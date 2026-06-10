#!/usr/bin/env python3
"""VGDL generation harness that sweeps every context level for multiple games.

The script performs the following high-level steps for a requested generation model:
    1. Enumerates every observation (first/last 10, iterations 0-4, all games).
    2. Builds five tiers of context (levels 0-4) with shared VGDL grammar data.
    3. Runs both the non-SCM and SCM tasks for *each* level via dedicated
       functions, emitting transformer-generated VGDL drafts.
    4. Randomly produces partial VGDL snippets by removing rules from the
       ground-truth VGDL files up to a configurable threshold.
    5. Stores only the generated VGDLs in a JSON file under
       `VGDL_gen/<sanitized_model_name>.json`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import random
import textwrap
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import re

import torch
from utils import _generate_from_messages, load_model_tokenizer, unload, _separate_thinking


# -----------------------------------------------------------------------------
# Paths and shared configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
OBSERVATION_ROOT = REPO_ROOT / "observations"
VGDL_OUTPUT_ROOT = REPO_ROOT / "VGDL_gen"
GVG_GAME_ROOT = REPO_ROOT / "GVGAI_GYM" / "gym_gvgai" / "envs" / "games"

# Restricted to only use the 'first_10' folder.
VALID_SLICES = ("first_10", "last_10")
# Select up to 5 iterations.
ITERATION_RANGE = range(num_iterations := 5)
VGDL_SECTIONS = ("BasicGame", "SpriteSet", "InteractionSet", "TerminationSet")


def ensure_directory(path: Path) -> None:
    """Create `path` (and its parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_unique_json_path(path: Path) -> Path:
    """Return a non-conflicting JSON path (adds timestamp suffix if needed)."""
    if not path.exists():
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{path.suffix}")


# Fill this map with real descriptions when available.
GAME_DESCRIPTION_OVERRIDES: Dict[str, str] = {
    "racebet": "The objective of this game is guessing which camel is going to win the race. Betting is performed by placing the player in one of the corresponding betting spots. No points are given in this game. You either bet for the one that wins (and the player wins) or you don't (and the player loses).",
    "digdug": "The player must collect all gems and gold coins in the cave, digging its way through it, to win the game. There are also enemies in the level that kill the player on collision with them. Also, the player can shoot boulders by pressing USE two consecutive time steps, which kill enemies.",
    "brainman": "The player must push keys into doors to open them, to reach the exit to win the game. Many gems around are available for collection, which give score to the player. Keys are pushed by colliding with them, and they are propelled in a straight line until finding a sprite to collide against, when they stop. If they hit a door, both the door and the key are destroyed.",
    "defender": "The objective of this game is to avoid that the enemies destroy your city. The enemies are spawned at the right of the screen and cross it while bombing the city. The player must eliminate these enemies as soon as possible so the city is not destroyed, otherwise the player loses. The player’s ammunition is supplied from the top of the screen and must be collected periodically to avoid running out of ammo.",
    "lasers2": "The objective of this game is to reach the exit, which is typically blocked by a mud sprite. The mud can be destroyed by directing a laser to it. There are several laser cannons whose orientation can be changed by the player shooting at them. Apart from this, there are several crystal blocks, which can also be rotated, that deflect the lasers in a certain direction. Lasers kill the player when in contact.",
    "boulderchase": "In this game, the player must dig inside a cave to collect at least 9 gems and reach the exit. As the player digs, boulders that are left without a ground fall and can kill the player upon collision. Enemies also move around the cave, being able to dig as well. When digging, they drop gems that can also be collected by the player. Enemies are also killed by boulders and kill the player upon contact.",
    "portals": "The player controls an avatar that needs to find the exit of a maze. The maze is full of entry and exit portals, and every time the player goes through an entry portal they will be teleported to an exit portal at random. The maze is also full of enemies and missiles that kill the player on contact.",
    "iceandfire": "The objective of this game is to find the exit of the maze. The maze has multiple traps, two different types of surfaces (ice and fire), and two types of boots (for ice and fire) that can be collected. The special surfaces kill the player on contact, unless the appropriate boots are picked up beforehand. There are several coins scattered around the level that give the player some score.",
    "cakybaky": "The objective of the game is to bake a cake. There are several ingredients that must be collected in order, to follow the recipe. This order to follow is flour, milk, eggs, sugar, butter and cherries, although not all levels have all these ingredients. There are angry chefs around the level that chase the player, although they only care about their favourite ingredient, so only the ones that prefer the next ingredient to be picked up are active at each time. If a chef collides with the player, the player loses."
}


def get_game_description(game_name: str) -> str:
    """Fetch a game description directly from the overrides dictionary."""
    return GAME_DESCRIPTION_OVERRIDES.get(game_name, f"Add description for {game_name}")

COMMON_VGDL_GRAMMAR = textwrap.dedent(
    """
    A game is defined by two separate components, the level description, which essentially describes the
    positions of all objects and the layout of the game in 2D (i.e., the initial conditions), and the game
    description proper, which describes the dynamics and potential interactions of all the objects in the game.
    The level description is simply a text string/file with a number of equal-length lines, where each
    character maps to (read: instantiates) one or more objects at the corresponding location of the
    rectangular grid.
    
    The game description i.e. the VGDL is composed of four blocks of instructions.
    
    The LevelMapping describes how to translate the characters in the level description into (one or more)
    objects, to generate the initial game state. For example in a certain game, each "1" spawns an object of the "monster" class.
    
    The SpriteSet defines the classes of objects used, all of which are defined in the ontology, and derive
    from an abstract VGDLSprite class. Object classes are organized in a tree (using nested indentations),
    where a child class will inherit the properties of its ancestors. For example, there are two subclasses of
    avatars, one where Link possesses the key and one where he does not. Furthermore, all class definitions
    can be augmented by keyword options. For example, the "key" and "goal" classes differ only by their color
    and how they interact.
    
    The InteractionSet defines the potential events that happen when two objects collide. Each such
    interaction maps two object classes to an event method (defined in the ontology), possibly augmented by
    keyword options. For example, swords kill monsters, monsters kill the avatar (both subclasses), nobody is
    allowed to pass through walls, and when Link finds a "key" object, the avatar class is transformed.
    
    The TerminationSet defines different ways by which the game can end, each line is a termination criterion,
    different criteria are available through the ontology and can be further specialized with keyword options.
    Here, it is sufficient to capture the goal (bring the sprite counter of the "goal" class to zero) to win.
    What permits the descriptions to be so concise is an underlying ontology which defines many high-level
    building blocks for games, including the types of physics used (continuous, or grid based, friction,
    gravity, etc.), movement dynamics of objects (straight or random motion, player-control, etc.) and
    interaction effects upon object collisions (bouncing, destruction, spawning, transformation, etc.).
    
    Here is the grammar:
    <game> ::= game_class <eol> INDENT <level-block> <sprite-block> <interaction-block> <termination-block>
    <level-block> ::= LevelMapping <eol> INDENT { <char-map> NEWLINE } DEDENT
    <sprite-block> ::= SpriteSet <eol> INDENT { <sprite-def> NEWLINE } DEDENT
    <interaction-block> ::= InteractionSet <eol> INDENT { <interaction-def> <eol> } DEDENT
    <termination-block> ::= TerminationSet <eol> INDENT { <termination-def> <eol> } DEDENT
    <char-map> ::= CHAR " > " <sprite-type> { " " <sprite-type> }
    <sprite-def> ::= <sprite-simple> [ <eol> INDENT { <sprite-def> <eol> } DEDENT ]
    <sprite-simple> ::= <sprite-type> " > " [ sprite class ] { " " <option> }
    <interaction-def> ::= <sprite-type> <sprite-type> " > " interaction method { " " <option> }
    <termination-def> ::= termination class { " " <option> }
    <eol> ::= { " " } [ "#" { CHAR | " " } ] NEWLINE
    <option> ::= IDENTIFIER "=" ( <sprite-type> | evaluable )
    <sprite-type> ::= IDENTIFIER | "avatar" | "wall" | "EOS"
    """
).strip()

EXAMPLE_VGDL_CODE = textwrap.dedent("""
    BasicGame
        SpriteSet
            floor > Immovable img=newset/floor6 hidden=True
            human >
                annoyed > RandomNPC speed=0.25 img=newset/cursedman cons=2
                citizen >
                    quiet > RandomNPC speed=0.25 img=newset/man2 cons=1
                    avatar > ShootAvatar stype=cigarette img=newset/girl1 rotateInPlace=False
            george > Chaser stype=citizen speed=0.25 img=newset/man4 frameRate=8
            cigarette > Flicker limit=5 singleton=True img=newset/cigarette
            wall > Immovable img=oryx/wall6

        TerminationSet
            SpriteCounter stype=avatar  win=False
            SpriteCounter stype=quiet   win=False
            Timeout limit=1000 win=True

        InteractionSet
            quiet george > transformTo stype=annoyed
            avatar george > killSprite scoreChange=-1
            annoyed cigarette > transformTo stype=quiet scoreChange=1
            human wall wall > stepBack

        LevelMapping
            g > floor george
            c > floor quiet
            A > floor avatar
            . > floor
    """).strip()

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


class ContextLevel(IntEnum):
    OBS_ONLY = 0          # Baseline: grid based observations only
    OBS_WITH_GRAMMAR = 1  # Syntax Check: grid based observations + Grammar + Example
    EXACT_GAME = 2        # Grounding: Level 1 + Specific Game Rules
    GAMES_LIST = 3        # Robustness: Level 1 + List of Games (Distractors)
    PARTIAL_VGDL = 4      # Completion: Level 2 + Incomplete Code

    @classmethod
    def from_int(cls, raw: int) -> "ContextLevel":
        try:
            return cls(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Unsupported context level: {raw}") from exc


@dataclass
class ObservationSelection:
    """Describes where to fetch the observation text from."""
    slice_name: str
    iteration: int
    game_file: str

    def filepath(self) -> Path:
        return OBSERVATION_ROOT / self.slice_name / str(self.iteration) / self.game_file


@dataclass
class ObservationPayload(ObservationSelection):
    """Observation selection plus the loaded text."""
    text: str = ""

    def preview(self, max_chars: int = 100000) -> str:
        snippet = self.text.strip().replace("\n\n", "\n")
        return snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")


@dataclass
class GameDescriptor:
    """Metadata placeholder for an individual game."""
    name: str
    notes: str = "Notes pending—add distinguishing traits when available."

    @property
    def description(self) -> str:
        return get_game_description(self.name)

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description, "notes": self.notes}


@dataclass
class VGDLContext:
    """Aggregated context handed to prompt builders."""
    level: ContextLevel
    observation: ObservationPayload
    games_catalog: List[GameDescriptor] = field(default_factory=list)
    target_game: Optional[GameDescriptor] = None
    vgdl_grammar_rules: str = ""
    example_vgdl: str = ""
    partial_vgdl: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["level"] = int(self.level)
        data["games_catalog"] = [game.to_dict() for game in self.games_catalog]
        if self.target_game is not None:
            data["target_game"] = self.target_game.to_dict()
        return data


@dataclass
class SCMBlueprint:
    nodes: Dict[str, str]
    edges: List[Sequence[str]]
    vgdl_notes: List[str]
    scm_notes: List[str] = field(default_factory=list)
    allow_extra: bool = True

    def to_dict(self) -> Dict:
        return {
            "nodes": self.nodes,
            "edges": [list(edge) for edge in self.edges],
            "vgdl_notes": self.vgdl_notes,
            "scm_notes": self.scm_notes,
        }

    def to_text(self) -> str:
        parts = ["Nodes:"] + [f"- {k}: {v}" for k, v in self.nodes.items()]
        parts += ["\nEdges:"] + [f"- {s}->{t}" for s, t in self.edges]
        if self.scm_notes: parts += ["\nSCM Authoring Rules:"] + [f"- {n}" for n in self.scm_notes]
        parts += ["\nVGDL Compiling Rules:"] + [f"- {n}" for n in self.vgdl_notes]
        if self.allow_extra: parts.append("\nExtra context-dependent nodes allowed.")
        return "\n".join(parts)


# -----------------------------------------------------------------------------
# Partial VGDL generator
# -----------------------------------------------------------------------------


def read_ground_truth_vgdl(game_name: str) -> Optional[str]:
    """Load the canonical VGDL text for a given game if available."""
    path = GVG_GAME_ROOT / f"{game_name}_v0" / f"{game_name}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")



def generate_partial_vgdl_from_ground_truth(
    game_name: str,
    removal_threshold: int, # Kept for signature compatibility, but unused
    seed: Optional[int] = None, # Kept for signature compatibility, but unused
    gt_text: Optional[str] = None,
) -> str:
    """
    Create a partial VGDL by:
    1. Keeping SpriteSet and LevelMapping 100% intact.
    2. Completely removing InteractionSet and TerminationSet.
    """
    gt_path = GVG_GAME_ROOT / f"{game_name}_v0" / f"{game_name}.txt"
    # Assuming read_ground_truth_vgdl is available in your scope
    source_text = gt_text if gt_text is not None else read_ground_truth_vgdl(game_name)
    
    if source_text is None:
        return f"# Missing GT VGDL for {game_name}: {gt_path}"

    lines = source_text.splitlines()
    # We use a list that allows None so we can mark lines for deletion
    processed_lines: List[Optional[str]] = list(lines) 

    def collect_section_indices(section: str) -> List[int]:
        """Collects indices of the BODY of a section (excluding header)."""
        indices: List[int] = []
        capturing = False
        header_indent = 0
        
        for idx, line in enumerate(processed_lines):
            if line is None:
                continue
                
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            if capturing:
                if not stripped:
                    indices.append(idx)
                    continue
                if indent > header_indent:
                    indices.append(idx)
                    continue
                capturing = False
            
            if stripped.startswith(section):
                capturing = True
                header_indent = indent
                
        return indices

    def remove_whole_section(section: str) -> None:
        """Removes the section header and its entire body."""
        # 1. Get the body lines
        body_indices = collect_section_indices(section)
        
        # 2. Find the header line manually
        header_idx = -1
        for idx, line in enumerate(processed_lines):
            if line is not None and line.strip().startswith(section):
                header_idx = idx
                break
        
        # 3. Mark both header and body for deletion
        if header_idx != -1:
            processed_lines[header_idx] = None
        
        for idx in body_indices:
            processed_lines[idx] = None

    # --- Execution ---

    # Completely remove InteractionSet and TerminationSet
    remove_whole_section("InteractionSet")
    remove_whole_section("TerminationSet")

    # SpriteSet and LevelMapping are untouched and remain in 'processed_lines'

    header = [
        f"# Partial VGDL for {game_name}",
        "# InteractionSet and TerminationSet removed.",
        "# SpriteSet and LevelMapping preserved.",
    ]
    
    # Filter out lines marked as None before joining
    final_output = [line for line in processed_lines if line is not None]

    return "\n".join(header + final_output)


def build_non_scm_prompt(context: VGDLContext) -> str:
    """Return a prompt template for direct VGDL generation with CoT enforcement."""
    context_json = json.dumps(context.to_dict(), indent=2)

    # Instructions based on new Information Density Hierarchy
    if context.level == ContextLevel.OBS_ONLY:
        task_specifics = (
            "Analyze the <observation_log> which contains text-based 2D grids (e.g., rows of 'wall', 'avatar', etc.). "
            "Infer collision dynamics and sprite physics solely from how these grids change over time."
        )
    elif context.level == ContextLevel.OBS_WITH_GRAMMAR:
        task_specifics = (
            "Analyze the <observation_log> grids. Use the provided <vgdl_grammar> and the <example_vgdl> "
            "to ensure your syntax is correct. Do not copy the example logic, only its syntactic structure."
        )
    elif context.level == ContextLevel.EXACT_GAME:
        task_specifics = (
            "You have the game description of the <exact_game>. Map the rules described in the text "
            "to the visual entities found in the <observation_log> grids."
        )
    elif context.level == ContextLevel.GAMES_LIST:
        task_specifics = (
            "You have a list of candidate games. Identify which game matches the entities seen in the <observation_log> grids, "
            "then generate the VGDL for that specific game."
        )
    elif context.level == ContextLevel.PARTIAL_VGDL:
        task_specifics = (
            "You have a partial VGDL file. Use the <observation_log> grids to reverse-engineer the missing interactions "
            "and complete the code."
        )
    else:
        task_specifics = "Generate VGDL based on the context."

    prompt = textwrap.dedent(f"""
    You are a VGDL (Video Game Description Language) expert.
    
    ### Task
    {task_specifics}
    Your goal is to write a complete, executable VGDL game description.

    ### Input Data
    <context>
    {context_json}
    </context>

    ### Observation Format Note
    The observations are provided as a sequence of text grids.
    Each grid represents the game board at a specific time step.
    - 'wall', 'avatar', 'green', 'key', etc. represent objects at that coordinate.
    - Changes in coordinates between steps indicate movement.
    - Objects disappearing or changing types indicate interactions.

    ### Instructions
    1. **Analysis Phase**: Before writing code, briefly list the objects (sprites) you detect in the grids and how they interact (e.g., "The 'avatar' moves onto 'key', and 'key' disappears").
    2. **Synthesize VGDL**: Write the code blocks: `SpriteSet`, `LevelMapping`, `InteractionSet`, `TerminationSet`.
    
    ### Constraints
    - Do not invent sprites not present in the observation grids.
    - Ensure every symbol in `LevelMapping` is defined in `SpriteSet`.
    - STRICTLY output the VGDL code inside a single code block.
    """)
    return prompt


def build_scm_prompt(context: VGDLContext, blueprint: SCMBlueprint) -> str:
    """
    Return a prompt for generating a Mechanics-Oriented SCM.
    Focuses on 'Precondition -> Interaction -> Postcondition' logic.
    """
    context_json = json.dumps(context.to_dict(), indent=2)
    
    # SCM specific instructions based on context level
    if context.level == ContextLevel.OBS_ONLY:
        level_instruction = "Infer causal relationships purely by tracking object movements and collisions in the text grids."
    elif context.level == ContextLevel.OBS_WITH_GRAMMAR:
        level_instruction = "Infer causal relationships from the grids. Use the <vgdl_grammar> and <example_vgdl> to name your entities and interactions correctly."
    elif context.level == ContextLevel.EXACT_GAME:
        level_instruction = "Correlate the rules in the <target_game> description with the events seen in the text grids to build the causal model."
    elif context.level == ContextLevel.GAMES_LIST:
        level_instruction = "Determine which game from the list matches the grid events, and build a causal model for that specific game."
    elif context.level == ContextLevel.PARTIAL_VGDL:
        level_instruction = "Use the <partial_vgdl> to identify known entities, and fill in the missing causal logic using the text grids."
    else:
        level_instruction = "Build a causal model from the context."

    prompt = textwrap.dedent(f"""
    You are a Game Mechanics Reverse-Engineer.
    Your task is to build a Structural Causal Model (SCM) of the game mechanics based on the provided logs.

    ### Context
    <context>
    {context_json}
    </context>

    ### Task
    {level_instruction}

    ### Observation Format
    The logs contain sequences of 2D text grids. 
    You must identify entities (e.g., 'avatar', 'wall', 'blue') by their string labels in these grids.

    ### SCM Structural Requirements
    Follow this blueprint strictly to define the causal graph:
    
    {blueprint.to_text()}

    ### Output Format
    Provide the SCM as a structured JSON object inside a code block so it can be parsed programmatically.
    """)
    return prompt


def build_vgdl_from_scm_prompt(
    context: VGDLContext,
    scm_text: str,
    blueprint: SCMBlueprint,
) -> str:
    """Prompt template for producing VGDLs from SCM reasoning artifacts."""
    
    context_json = json.dumps(context.to_dict(), indent=2)

    instructions = textwrap.dedent("""
    You are a VGDL Compiler. 
    Your input is a Structural Causal Model (SCM) describing game mechanics, and the original context.
    Your output is the executable VGDL code.

    ### Source Material
    
    **1. Context (The Raw Data):**
    {context}
    
    **2. Causal Model (The Logic derived from Context):**
    {scm}

    **3. Blueprint (The Structure):**
    {blueprint}

    ### Compilation Rules
    - **SpriteSet**: Map the 'Entity Identification' from SCM to VGDL sprite types (e.g., `Immovable`, `Missile`, `Flicker`).
    - **InteractionSet**: Translate 'Interaction Dynamics' directly into VGDL interactions.
        - `[Avatar] hits [Wall] -> [StepBack]` becomes `avatar wall > stepBack`
    - **LevelMapping**: Use the symbols defined in the blueprint.
    - **TerminationSet**: Derive win/loss conditions from 'Termination Mechanics' in the SCM.

    ### Output
    Provide ONLY the VGDL code block.
    """).format(
        context=context_json,
        scm=scm_text,
        blueprint=blueprint.to_text()
    )
    return instructions


# -----------------------------------------------------------------------------
# Context builders (Refactored for new ordering and cumulative data)
# -----------------------------------------------------------------------------


def build_context_level_0(observation: ObservationPayload) -> VGDLContext:
    # OBS_ONLY
    return VGDLContext(level=ContextLevel.OBS_ONLY, observation=observation, games_catalog=[])


def build_context_level_1(
    observation: ObservationPayload
) -> VGDLContext:
    # OBS_WITH_GRAMMAR: Level 0 + Grammar + Example
    context = VGDLContext(level=ContextLevel.OBS_WITH_GRAMMAR, observation=observation, games_catalog=[])
    context.vgdl_grammar_rules = COMMON_VGDL_GRAMMAR
    context.example_vgdl = EXAMPLE_VGDL_CODE
    return context


def build_context_level_2(
    observation: ObservationPayload,
    target: GameDescriptor,
) -> VGDLContext:
    # EXACT_GAME: Level 1 + Target Game
    # Ensures previous context (Grammar + Example) is present
    context = build_context_level_1(observation)
    context.level = ContextLevel.EXACT_GAME
    context.target_game = target
    return context


def build_context_level_3(
    observation: ObservationPayload,
    games_catalog: List[GameDescriptor],
) -> VGDLContext:
    # GAMES_LIST: Level 1 + Games List
    # Ensures previous context (Grammar + Example) is present
    context = build_context_level_1(observation)
    context.level = ContextLevel.GAMES_LIST
    context.games_catalog = games_catalog
    return context


def build_context_level_4(
    observation: ObservationPayload,
    target: GameDescriptor,
    partial_vgdl: str,
) -> VGDLContext:
    # PARTIAL_VGDL: Level 2 + Partial Code
    # Ensures previous context (Grammar + Example + Target Game) is present
    context = build_context_level_2(observation, target)
    context.level = ContextLevel.PARTIAL_VGDL
    context.partial_vgdl = partial_vgdl
    return context


LEVEL_BUILDERS = {
    ContextLevel.OBS_ONLY: build_context_level_0,
    ContextLevel.OBS_WITH_GRAMMAR: build_context_level_1,
    ContextLevel.EXACT_GAME: build_context_level_2,
    ContextLevel.GAMES_LIST: build_context_level_3,
    ContextLevel.PARTIAL_VGDL: build_context_level_4,
}


# -----------------------------------------------------------------------------
# Task runners (explicit functions for SCM / non-SCM)
# -----------------------------------------------------------------------------


def run_non_scm_task(context: VGDLContext, generate_text: Callable[[str], str]) -> str:
    """Generate a VGDL directly from the provided context."""
    prompt = build_non_scm_prompt(context)
    return generate_text(prompt)


def run_scm_task(context: VGDLContext, gen_text: Callable[[str], str]) -> Tuple[str, str]:
    bp = SCMBlueprint(
        nodes={
            # Design Layer
            "EntityTypes": "Static sprite types, roles (avatar, wall), & attrs (speed, hp). Not time-varying.",
            "ActionSpace": "Agent interventions/actions available at each step (move, shoot).",
            "GlobalMechanics": "Global state transitions independent of collisions (gravity, timers).",
            "InteractionMechanics": "Collision rules mapping (Entities, State_t, Action_t) -> State_{t+1} events (spawn, destroy).",
            "RewardMechanics": "Maps State_t/Interactions to Reward_t or Score_{t+1}.",
            "TerminationMechanics": "Win/loss conditions based on StateVariables (e.g., timeout, goals_met).",
            # Dynamics Layer
            "StateVariables": "Dynamic vars evolving over t (pos, hp). Eq: X_{t+1}=f(PA_X_t, Action_t, U_X).",
            "InitialState": "StateVariables at t=0 derived from EntityTypes + LevelEncoding.",
            # Observation Layer
            "LevelEncoding": "ASCII grid mapping chars to entity instances/positions; induces InitialState.",
        },
        edges=[
            ("EntityTypes", "InitialState"), ("LevelEncoding", "InitialState"), ("EntityTypes", "StateVariables"),
            ("ActionSpace", "InteractionMechanics"), ("ActionSpace", "StateVariables"),
            ("GlobalMechanics", "StateVariables"), ("InteractionMechanics", "StateVariables"),
            ("StateVariables", "InteractionMechanics"), ("StateVariables", "RewardMechanics"),
            ("InteractionMechanics", "RewardMechanics"), ("RewardMechanics", "StateVariables"),
            ("StateVariables", "TerminationMechanics"), ("InteractionMechanics", "TerminationMechanics"),
        ],
        scm_notes=[
            "Define DYNAMIC SCM with discrete time steps t=0,1...",
            "Separate STATIC design vars from DYNAMIC state vars.",
            "For dynamic X, define X_{t+1} = f(Parents_t, Action_t, Noise).",
            "Define Reward_t & Termination explicitly as equations.",
            "Output machine-readable format (JSON) with: static_nodes, dynamic_variables, edges, equations.",
            "Link LevelEncoding chars to InitialState.",
        ],
        vgdl_notes=[
            "EntityTypes+StateVariables -> VGDL SpriteSet (roles, attrs).",
            "LevelEncoding+InitialState -> VGDL LevelMapping.",
            "InteractionMechanics+GlobalMechanics -> VGDL InteractionSet.",
            "RewardMechanics+TerminationMechanics -> VGDL TerminationSet.",
            "ActionSpace -> VGDL Avatar controls.",
        ],
    )

    scm_text = gen_text(build_scm_prompt(context, bp))
    vgdl_text = gen_text(build_vgdl_from_scm_prompt(context, scm_text, bp))

    return vgdl_text, scm_text


def extract_vgdl_and_after(text):
    """
    Extracts VGDL code with a robust fallback strategy.
    
    Priority:
    1. ```vgdl ... ``` block
    2. ``` ... ``` block
    3. Heuristic: Text starting from VGDL keywords (BasicGame, SpriteSet, etc.)
    4. Fallback: The entire raw text (so JSON is never empty).
    
    Returns: (vgdl_code, context)
    """
    if not text:
        return "", ""

    # Priority 1: Explicit ```vgdl
    pattern_explicit = r'(.*?)```vgdl(.*?)```(.*)$'
    match = re.search(pattern_explicit, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(2).strip(), (match.group(1) + "\n" + match.group(3)).strip()

    # Priority 2: Generic ```
    pattern_generic = r'(.*?)```(.*?)```(.*)$'
    match = re.search(pattern_generic, text, re.DOTALL)

    if match:
        return match.group(2).strip(), (match.group(1) + "\n" + match.group(3)).strip()

    # Priority 3: Heuristic scan for VGDL keywords
    # The code might not start with BasicGame (e.g. partial generation).
    # We look for the earliest occurrence of major VGDL blocks.
    keywords = ["BasicGame", "SpriteSet", "LevelMapping", "InteractionSet", "TerminationSet"]
    
    earliest_idx = -1
    
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1:
            if earliest_idx == -1 or idx < earliest_idx:
                earliest_idx = idx
                
    if earliest_idx != -1:
        # Split at the found keyword
        # Context is everything before, code is everything from the keyword onwards
        return text[earliest_idx:].strip(), text[:earliest_idx].strip()

    # Priority 4: Last Resort - Return raw text
    # This ensures your JSON file is NOT empty, allowing you to debug the LLM output.
    return text.strip(), "RAW_OUTPUT_NO_FORMATTING_FOUND"


# -----------------------------------------------------------------------------
# Core generator
# -----------------------------------------------------------------------------


class VGDLGenerator:
    """High-level orchestration for building contexts and storing artifacts."""

    def __init__(
        self,
        observation_root: Path = OBSERVATION_ROOT,
        output_root: Path = VGDL_OUTPUT_ROOT,
        generation_model_name: str = "gpt2",
        generation_device: Optional[int] = None,
        max_new_tokens: int = 20000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        # Quantize flag
        quantize: bool = False,
    ) -> None:
        self.observation_root = observation_root
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.generation_model_name = generation_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.quantize = quantize
        self._quantization_option = "4bit-nf4" if quantize else None
        print(f"[TRACE] Configured for model: {generation_model_name}")
        
        self.model = None
        self.tokenizer = None
        self._games_cache: Optional[List[GameDescriptor]] = None

    def _load_generation_model(self) -> None:
        """Explicitly load the generation model into VRAM."""
        print(f"[TRACE] Loading generation model via utils: {self.generation_model_name}")
        self.model, self.tokenizer = load_model_tokenizer(
            self.generation_model_name,
            quantization=self._quantization_option,
        )
        if (
            self.tokenizer is not None
            and self.tokenizer.pad_token_id is None
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _unload_generation_model(self) -> None:
        """Unload the generation model to free up VRAM."""
        print("[TRACE] Unloading generation model...")
        if self.model is not None:
            unload(self.model)
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_text_with_limit(self, prompt: str, max_tokens: int) -> str:
        """Core generator helper that invokes the model with a token budget."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Generation model is not loaded. Call _load_generation_model() first.")
        messages = [{"role": "user", "content": prompt}]
        temp = _generate_from_messages(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_tokens,
            )
        vgdl, _ = _separate_thinking(temp)
        return vgdl

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the configured max_new_tokens budget."""
        return self._generate_text_with_limit(prompt, self.max_new_tokens)

    def _generate_text_extended(self, prompt: str) -> str:
        """Generate text with an extra 10,000 token budget for tougher prompts."""
        return self._generate_text_with_limit(prompt, self.max_new_tokens + 10000)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_full_generation(
        self,
        removal_threshold: int,
        seed: Optional[int] = None,
        num_iterations: int = 1,
    ) -> Dict:
        """Sweep all slices, iterations, games, and context levels."""
        
        # Start Timing the generation process
        start_gen_time = time.time()

        # Derive identifier from model name
        sanitized_name = self.generation_model_name.replace("/", "_")
        # Append 4bit suffix if quantized
        if self.quantize:
            sanitized_name += "_4bit"

        games_catalog = self._discover_games_catalog()
        observation_index = self._build_observation_index()
        rng = random.Random(seed)

        aggregated: Dict[str, Dict] = {"non_scm": {}, "scm": {}}

        # --- PHASE 1: GENERATION ---
        self._load_generation_model()

        try:
            for level in ContextLevel:
                print(f"[TRACE] Processing context level {level.name} ({int(level)})")
                level_key = f"level_{int(level)}"
                aggregated["non_scm"].setdefault(level_key, {})
                aggregated["scm"].setdefault(level_key, {})
                for slice_name, iterations in observation_index.items():
                    print(f"[TRACE]  Slice '{slice_name}' with {len(iterations)} iteration(s)")
                    aggregated["non_scm"][level_key].setdefault(slice_name, {})
                    aggregated["scm"][level_key].setdefault(slice_name, {})
                    for iteration, game_files in sorted(iterations.items()):
                        if iteration >= num_iterations:
                            continue
                        iter_key = f"iteration_{iteration}"
                        print(f"[TRACE]   Iteration {iteration} covering {len(game_files)} game(s)")
                        aggregated["non_scm"][level_key][slice_name].setdefault(iter_key, {})
                        aggregated["scm"][level_key][slice_name].setdefault(iter_key, {})
                        for game_file in game_files:
                            print(f"[TRACE]     Generating VGDL for '{game_file}'")
                            selection = ObservationSelection(slice_name, iteration, game_file)
                            observation = self._load_observation(selection)
                            target_descriptor = self._match_game_descriptor(observation.game_file)
                            gt_vgdl_text = read_ground_truth_vgdl(target_descriptor.name)
                            partial_seed = rng.randint(0, 10_000_000)
                            partial_vgdl = generate_partial_vgdl_from_ground_truth(
                                target_descriptor.name,
                                removal_threshold=removal_threshold,
                                seed=partial_seed,
                                gt_text=gt_vgdl_text,
                            )
                            context = self._build_context_for_level(
                                level=level,
                                observation=observation,
                                games_catalog=games_catalog,
                                target=target_descriptor,
                                partial_vgdl=partial_vgdl,
                            )
                            
                            # Generate without scoring
                            non_scm_vgdl, non_scm_vgdl_context = extract_vgdl_and_after(run_non_scm_task(context, self._generate_text))
                            scm_vgdl, scm = run_scm_task(context, self._generate_text)
                            scm_vgdl, scm_vgdl_context = extract_vgdl_and_after(scm_vgdl
                            )
                            
                            if not non_scm_vgdl:
                                non_scm_vgdl, non_scm_vgdl_context = extract_vgdl_and_after(
                                    run_non_scm_task(context, self._generate_text_extended)
                                )
                            if not scm_vgdl:
                                scm_vgdl, scm = run_scm_task(context, self._generate_text_extended)
                                scm_vgdl, scm_vgdl_context = extract_vgdl_and_after(
                                scm_vgdl
                                )

                            # Store only the text, scores are None
                            aggregated["non_scm"][level_key][slice_name][iter_key][target_descriptor.name] = {
                                "vgdl": non_scm_vgdl,
                                "context": non_scm_vgdl_context,
                                "scores": None,
                            }
                            aggregated["scm"][level_key][slice_name][iter_key][target_descriptor.name] = {
                                "vgdl": scm_vgdl,
                                "context": scm_vgdl_context,
                                "scm": scm,
                                "scores": None,
                            }
        finally:
            self._unload_generation_model()

        # End timing
        end_gen_time = time.time()
        runtime_seconds = end_gen_time - start_gen_time

        payload = {
            "llm_name": self.generation_model_name,
            "generation_model": self.generation_model_name,
            "quantized_4bit": self.quantize,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "runtime_seconds": runtime_seconds,
            "results": aggregated,
        }
        output_path = self._save_results_json(sanitized_name, payload)
        payload["output_file"] = str(output_path)
        return payload

    def list_games(
        self,
        slice_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> List[str]:
        """Quickly list observation files for a given slice/iteration."""
        index = self._build_observation_index()
        if slice_name is None:
            return sorted({game for iterations in index.values() for games in iterations.values() for game in games})
        iterations = index.get(slice_name, {})
        if iteration is None:
            return sorted({game for games in iterations.values() for game in games})
        return iterations.get(iteration, [])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_context_for_level(
        self,
        level: ContextLevel,
        observation: ObservationPayload,
        games_catalog: List[GameDescriptor],
        target: GameDescriptor,
        partial_vgdl: str,
    ) -> VGDLContext:
        builder = LEVEL_BUILDERS[level]
        if level == ContextLevel.OBS_ONLY:
            return builder(observation)
        if level == ContextLevel.OBS_WITH_GRAMMAR:
            return builder(observation)
        if level == ContextLevel.EXACT_GAME:
            return builder(observation, target)
        if level == ContextLevel.GAMES_LIST:
            return builder(observation, games_catalog)
        if level == ContextLevel.PARTIAL_VGDL:
            return builder(observation, target, partial_vgdl)
        raise ValueError(f"Unhandled context level: {level}")

    def _discover_games_catalog(self) -> List[GameDescriptor]:
        if self._games_cache is not None:
            return self._games_cache
        names = set()
        for slice_name in VALID_SLICES:
            for iteration in ITERATION_RANGE:
                iter_dir = self.observation_root / slice_name / str(iteration)
                if not iter_dir.exists():
                    continue
                names.update(file.stem for file in iter_dir.glob("*.txt"))
        descriptors = [self._build_descriptor(name) for name in sorted(names)]
        self._games_cache = descriptors
        return descriptors

    def _build_descriptor(self, name: str) -> GameDescriptor:
        notes = "Notes pending—add distinguishing details when available."
        return GameDescriptor(name=name, notes=notes)

    def _match_game_descriptor(self, filename: str) -> GameDescriptor:
        name = Path(filename).stem
        for descriptor in self._discover_games_catalog():
            if descriptor.name == name:
                return descriptor
        return self._build_descriptor(name)

    def _load_observation(self, selection: ObservationSelection) -> ObservationPayload:
        path = selection.filepath()
        if not path.exists():
            available = sorted(p.name for p in path.parent.glob("*.txt"))
            raise FileNotFoundError(
                f"Observation file not found: {path}\nAvailable: {available}"
            )
        return ObservationPayload(
            slice_name=selection.slice_name,
            iteration=selection.iteration,
            game_file=selection.game_file,
            text=path.read_text(encoding="utf-8"),
        )

    def _build_observation_index(self) -> Dict[str, Dict[int, List[str]]]:
        index: Dict[str, Dict[int, List[str]]] = {}
        for slice_name in VALID_SLICES:
            slice_dir = self.observation_root / slice_name
            if not slice_dir.exists():
                continue
            iteration_map: Dict[int, List[str]] = {}
            for iteration in ITERATION_RANGE:
                iter_dir = slice_dir / str(iteration)
                if not iter_dir.exists():
                    continue
                files = sorted(file.name for file in iter_dir.glob("*.txt"))
                if files:
                    iteration_map[iteration] = files
            if iteration_map:
                index[slice_name] = iteration_map
        return index

    def _save_results_json(self, llm_name: str, data: Dict) -> Path:
        filename = f"{llm_name}_results.json"
        ensure_directory(self.output_root)
        target_path = ensure_unique_json_path(self.output_root / filename)
        payload = json.dumps(data, indent=2)
        try:
            target_path.write_text(payload, encoding="utf-8")
        except FileNotFoundError:
            ensure_directory(target_path.parent)
            target_path.touch()
            target_path.write_text(payload, encoding="utf-8")
        return target_path


# -----------------------------------------------------------------------------
# CLI wiring
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VGDL/SCM scaffolds for every context level and save results per LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-generation-model",
        required=True,
        help="Transformers model name or path used to generate VGDLs.",
    )
    parser.add_argument(
        "--generation-device",
        type=int,
        default=None,
        help="Device index for the transformers pipeline (-1 or omit for CPU).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32000,
        help="Maximum number of new tokens sampled during VGDL generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for the transformers generator.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter for the transformers generator.",
    )
    parser.add_argument(
        "--removal-threshold",
        type=int,
        default=2,
        help="Max number of SpriteSet/LevelMapping rules removed when creating partial VGDLs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic partial VGDLs.")
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="Print the available observation files and exit.",
    )
    parser.add_argument("--slice", choices=VALID_SLICES, help="Optional slice filter for --list-games.")
    parser.add_argument(
        "--iteration",
        type=int,
        choices=list(ITERATION_RANGE),
        help="Optional iteration filter for --list-games.",
    )
    # Control the number of iterations
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterations (folders 0-N) to process from the observations.",
    )
    # Control quantization
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable 4-bit NF4 quantization via bitsandbytes.",
    )
    return parser.parse_args()


def _main() -> None:
    start_time = time.time()
    args = _parse_args()
    generator = VGDLGenerator(
        generation_model_name=args.hf_generation_model,
        generation_device=args.generation_device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        quantize=args.quantize,
    )

    if args.list_games:
        games = generator.list_games(slice_name=args.slice, iteration=args.iteration)
        print("Observation files:")
        for game in games:
            print(f"- {game}")
        return

    payload = generator.run_full_generation(
        removal_threshold=args.removal_threshold,
        seed=args.seed,
        num_iterations=args.num_iterations,
    )
    output_path = payload.get("output_file", str(VGDL_OUTPUT_ROOT / f"{args.hf_generation_model.replace('/', '_')}_results.json"))
    print(f"Saved aggregated results to {output_path}")
    print(f"Entries recorded: non-SCM={len(payload['results']['non_scm'])}, SCM={len(payload['results']['scm'])}")
    
    end_time = time.time()
    print(f"[TRACE] Total runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    _main()