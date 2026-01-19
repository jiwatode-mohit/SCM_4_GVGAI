"""Utility helpers shared by the GVGAI evaluation scripts.

This module centralises prompt construction, model lifecycle helpers, metric
aggregation, and basic filesystem I/O routines.  By keeping these helpers in a
single place we can reuse them from notebooks, scripts, or CLI tools while
keeping the evaluation logic focused on orchestration rather than plumbing.
"""

from __future__ import annotations
import gc
import json
import re
import csv
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Mapping, Sequence
import warnings

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

# Centralised helper so progress tracing is consistent everywhere.
def log_progress(message: str) -> None:
    print(f"[TRACE] {message}", flush=True)

# ---------------------------------------------------------------------------
# Basic constants reused across jobs
# ---------------------------------------------------------------------------

new_games = True  # Whether to include the 10 new games



SYSTEM_RULES = (
    "You are an evaluator that must obey a strict output format.\n"
    "Line 1: ONLY the game name from the allowed list.\n"
    "Line 2: optional short reasoning.\n"
    "No extra lines, punctuation, or quotes."
)

# Example snippets used for VGDL from SCM prompts
EXAMPLE_SCM = """\
Variables
- PlayerPos: avatar x,y
- ThugPos[i]: thug positions
- BossAlive: {0,1}
- JailDoorOpen: {0,1}
- Keys: integer (0-1)
Structural Equations
- … (abridged for brevity) …
Causal Graph
- PlayerPos <- PlayerPos_prev, Keys
- BossAlive <- PlayerPos, BossAlive_prev
- …"""

EXAMPLE_VGDL = """\
BasicGame square_size=40
    SpriteSet
        floor > Immovable img=oryx/space4 hidden=True
        jaildoor    > Immovable    color=WHITE img=newset/jaildoor1
        depot > Immovable    color=WHITE img=oryx/key1
        jail    > Immovable    color=WHITE img=newset/jail
        transaction > Immovable invisible=True hidden=True
            redtrans >
            greentrans >
            yellowtrans >
        thug > Fleeing stype=avatar speed=0.9
            redthug > img=newset/redthug frameRate=7
            greenthug > img=newset/greenthug frameRate=9
            yellowthug > img=newset/yellowthug frameRate=5
            boss > img=newset/bossthug frameRate=12
        key > Resource limit=1 img=key singleton=True color=LIGHTBLUE
        villain > Resource
            redvillain > color=RED
            greenvillain > color=GREEN
            yellowvillain > color=YELLOW
        avatar > MovingAvatar img=newset/cop2 frameRate=8
        wall > Immovable img=oryx/wall1
    LevelMapping
        . > floor
        0 > floor jail
        1 > floor depot
        d > floor jaildoor
        g > floor greenthug
        y > floor yellowthug
        r > floor redthug
        b > floor boss
        A > floor avatar
    InteractionSet
        avatar  wall  > stepBack
        thug wall jaildoor > stepBack
        key avatar      > collectResource
        key avatar      > killSprite
        jaildoor avatar > killIfOtherHasMore resource=key limit=1
        avatar jaildoor > changeResource resource=key value=-1
        avatar jaildoor > stepBack
        avatar depot > spawnIfHasLess resource=key limit=0 stype=key
        avatar redthug > changeResource resource=redvillain value=1 killResource=True
        avatar greenthug > changeResource resource=greenvillain value=1  killResource=True
        avatar yellowthug > changeResource resource=yellowvillain value=1 killResource=True
        avatar jail > spawnIfHasMore resource=greenvillain limit=1 stype=greentrans
        avatar greentrans > changeResource resource=greenvillain value=-1 killResource=True scoreChange=2
        avatar jail > spawnIfHasMore resource=yellowvillain limit=1 stype=yellowtrans
        avatar yellowtrans > changeResource resource=yellowvillain value=-1 killResource=True scoreChange=5
        avatar jail > spawnIfHasMore resource=redvillain limit=1 stype=redtrans
        avatar redtrans > changeResource resource=redvillain value=-1 killResource=True scoreChange=100
        boss avatar > killSprite scoreChange=100
    TerminationSet
        Timeout limit=1900 win=False
        SpriteCounter      stype=boss               limit=0 win=True
"""


descriptions = {
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
required_games  = list(descriptions.keys())
AVAILABLE_GAMES = list(descriptions.keys())
VALID_GAMES = set(AVAILABLE_GAMES)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

DESCRIPTION_SEPARATOR = "End of game description\n----------------------------------------\n"


def strip_description_prefix(text: str) -> str:
    """Remove the prepended description section from cached observations."""
    log_progress("strip_description_prefix: inspecting observation text")
    if not text:
        return text
    if DESCRIPTION_SEPARATOR in text:
        _, remainder = text.split(DESCRIPTION_SEPARATOR, 1)
        return remainder.lstrip("\n")
    return text


def build_prompt_standard(observation: str, description_map: Optional[Dict[str, str]] = None) -> str:
    """Construct the default classification prompt with all game descriptions."""
    log_progress("build_prompt_standard: constructing classification prompt")
    description_source = description_map if description_map is not None else descriptions
    # Enumerate the available games so the model can quote either name or index
    num_list = "\n".join(f"{i}: {g}" for i, g in enumerate(AVAILABLE_GAMES))
    # Compose all game descriptions
    all_descriptions = "Here are the descriptions of all the available games, the observations later are from gameplay of one of these games:\n\n"
    for g in AVAILABLE_GAMES:
        # Fall back to a default text if we somehow lack a description override
        desc = description_source.get(g, "No description available.")
        all_descriptions += f"{g}:\n{desc}\n"
    all_descriptions += "End of game description\n" + "-" * 40 + "\n"
    return (
        "You are given 10 consecutive ASCII-rendered game screens produced by random actions in a single game. "
        "From their style, symbols, layout, and recurring patterns, deduce which game they come from. "
        "Choose exactly one title from the list below.\n\n"
        "Available games:\n" + num_list + "\n\n"
        + all_descriptions +
        "Answer format (strict):\n"
        "{ONLY the selected game name}.\n"
        "ASCII observations (10 frames):\n```text\n" + observation + "\n```"
    )

def build_prompt_vgdl(observation: str):
    """Return a VGDL-authoring chat prompt for VGDL generation jobs."""
    log_progress("build_prompt_vgdl: preparing VGDL prompt")
    system = (
        "You are an expert in Video Game Description Language (VGDL). "
        "Given a short description and a sequence of 10 gameplay observations with actions, "
        "infer the minimal VGDL specification."
    )
    # Chat-based prompts expect explicit role annotations
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": observation},
    ]

def build_prompt_natural(observation: str):
    """Return a natural-language explanation prompt for the observation."""
    log_progress("build_prompt_natural: preparing natural language prompt")
    system = (
        "You are an expert game-design analyst. "
        "Given a raw game state or description, produce a clear, concise natural language overview of the mechanics."
    )
    # Provide the observation directly as the user content
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": observation},
    ]

def build_prompt_scm(description: str):
    """Return a chat prompt that asks for a structural causal model."""
    log_progress("build_prompt_scm: preparing SCM prompt")
    system = (
        "You are a structural-causal-model (SCM) architect. "
        "Given a natural language game description, output a fully-specified SCM following the required sections." 
    )
    # Feed the description verbatim; the system instructions describe the format
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": description},
    ]

def build_prompt_vgdl_from_scm(scm: str):
    """Return a VGDL prompt seeded with an SCM-to-VGDL worked example."""
    log_progress("build_prompt_vgdl_from_scm: preparing VGDL-from-SCM prompt")
    system = (
        "You are a VGDL compiler. Convert a Structural Causal Model (SCM) into a minimal, syntactically-valid GVGAI VGDL file.\n"
        "\n### Example\nSCM:\n" + EXAMPLE_SCM + "\n\nCorresponding VGDL:\n" + EXAMPLE_VGDL + "\n### End of example\n\n"
        "Now convert the next SCM into VGDL, following the same style and guidelines above."
    )
    # Include the example in the system prompt so the model follows the format
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": scm},
    ]

VGDL_BASE_PATH = Path("GVGAI_GYM") / "gym_gvgai" / "envs" / "games"

@lru_cache(maxsize=None)
def load_vgdl_source(game_name: str) -> str:
    """Load the canonical VGDL script for ``game_name`` from disk."""
    log_progress(f"load_vgdl_source: loading VGDL for {game_name}")
    vgdl_dir = VGDL_BASE_PATH / f"{game_name}_v0"
    vgdl_file = vgdl_dir / f"{game_name}.txt"
    if not vgdl_file.exists():
        raise FileNotFoundError(f"VGDL file not found for game '{game_name}': {vgdl_file}")
    return vgdl_file.read_text(encoding="utf-8")


def build_prompt_vgdl_description(game_name: str, vgdl_source: str) -> List[Dict[str, str]]:
    """Prompt that summarises game mechanics from a VGDL definition."""
    log_progress(f"build_prompt_vgdl_description: preparing VGDL prompt for {game_name}")
    system = (
        "You are a VGDL analyst who specialises in translating GVGAI VGDL scripts into human-readable gameplay overviews. "
        "Explain only the core mechanics, objectives, sprite behaviors, interactions, and win/lose conditions. "
        "Do not copy code, Do not list every sprite, or Do not describe assets; stick to the gameplay rules and how they interact."
    )
    user = (
        "You are given the VGDL definition of a GVGAI game. Summarise the gameplay mechanics, objectives, and critical interactions "
        "without reproducing the VGDL text. Focus on how the player wins or loses and how the entities behave.\n\n"
        "The description should strictly not exceed 100 words\n\n"
        f"Game name: {game_name}\n\nVGDL definition:\n{vgdl_source}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_prompt_constructive_description(game_name: str, game_description: str) -> List[Dict[str, str]]:
    """Prompt that asks the model to *improve* an existing description."""
    log_progress(f"build_prompt_constructive_description: preparing constructive prompt for {game_name}")
    system = (
        "You are an assistant specialized in rewriting and enhancing technical descriptions of video games."
        "Your role is to take the provided input (game name + description) and produce a clearer, more structured explanation. "
        "You must focus on accuracy, clarity, and completeness of the description by breaking down gameplay mechanics, objectives, rules, and key features. "
        "Do not add persuasive, emotional, or marketing language. Do not invent details that are not present or logically implied in the original description."
        "The goal is to improve understanding, not to make the game sound more attractive."
    )
    user = ("You are given the name of a game and its description. Your task is to enhance the description by making it clearer, more detailed, and easier to understand." 
        "Focus on explaining the gameplay mechanics, objectives, and important features without adding marketing language or exaggeration." 
        "The goal is to create a more informative description that helps someone understand what the game is about."
        "The description should strictly not exceed 100 words\n\n"
        f"\n\nGame name: {game_name}\n\nOriginal description: {game_description}\n\n")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prompt_destructive_description(game_name: str, game_description: str) -> List[Dict[str, str]]:
    """Prompt that asks the model to create a fresh description from scratch."""
    log_progress(f"build_prompt_destructive_description: preparing destructive prompt for {game_name}")
    system = (
        "You are an assistant specialized in generating detailed and structured descriptions of video games."
        "Your role is to take the provided game name and produce a comprehensive explanation of the game "
        "based on your knowledge. You should cover gameplay mechanics, objectives, rules, and key features "
        "clearly and accurately. "
        "Do not add persuasive, emotional, or marketing language. "
        "If you are uncertain about specific details, give the most likely explanation based on common knowledge, "
        "but avoid fabricating overly specific or unverifiable claims."
    )
    user = (
        "You are given the name of a game. Your task is to generate a clear, detailed, and structured description "
        "of the game based on your knowledge. "
        "Focus on explaining the gameplay mechanics, objectives, and important features without exaggeration or "
        "marketing-style language."
        "The description should strictly not exceed 100 words\n\n"
        f"\n\nGame name: {game_name}\n\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prompt(job: str, observation: str, description_map: Optional[Dict[str, str]] = None):
    """Dispatch to the correct prompt builder for the selected job."""
    log_progress(f"build_prompt: dispatching prompt builder for job '{job}'")
    # Branch on the job name; chat-based jobs return role/content lists
    if job == "natural":
        return build_prompt_natural(observation)
    if job == "scm":
        return build_prompt_scm(observation)
    if job == "vgdl":
        return build_prompt_vgdl(observation)
    if job == "vgdl_from_scm":
        return build_prompt_vgdl_from_scm(observation)
    # Default to the classification prompt which returns a single string
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {
            "role": "user",
            "content": build_prompt_standard(
                observation, description_map=description_map
            ),
        },
    ]

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_quantization_config(option: Optional[str]) -> Optional[BitsAndBytesConfig]:
    """Construct a bitsandbytes quantization config for ``option`` if provided."""
    if option is None:
        return None

    normalized = option.lower()
    if normalized in {"", "none"}:
        return None
    log_progress(f"_build_quantization_config: requested option '{normalized}'")

    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if normalized in {"4bit", "4bit-fp4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if normalized in {"4bit-nf4", "nf4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    raise ValueError(f"Unsupported quantization option: {option}")


def load_model_tokenizer(model_name: str, quantization: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and tokenizer, handling provider-specific quirks."""
    log_progress(f"load_model_tokenizer: loading resources for {model_name}")
    quant_config = _build_quantization_config(quantization)
    model_id = model_name
    model_kwargs: Dict[str, object] = {}
    tokenizer_kwargs: Dict[str, object] = {}

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        # bitsandbytes quantization expects accelerate to map devices automatically
        model_kwargs.setdefault("device_map", "auto")

    if model_id.startswith("Qwen/") or model_id.startswith("deepseek-ai"):
        # Newer Qwen/DeepSeek families sometimes require remote code for custom tokenizers
        tokenizer_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True
        model_kwargs.setdefault("device_map", "auto")
        if "torch_dtype" not in model_kwargs and quant_config is None:
            model_kwargs["torch_dtype"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    elif model_id.startswith("tiiuae/Falcon3"):
        model_kwargs.setdefault("device_map", "auto")
        if "torch_dtype" not in model_kwargs and quant_config is None:
            model_kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_id.startswith("ibm-granite"):
        device = "cuda"
        # Granite checkpoints expect an explicit device string rather than "auto"
        model_kwargs["device_map"] = device
        if "torch_dtype" not in model_kwargs and quant_config is None:
            model_kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_id.startswith("meta"):
        if quant_config is None:
            model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        model_kwargs.setdefault("device_map", "auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        # Meta family typically benefits from bfloat16 to reduce VRAM usage
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    elif model_id.startswith("nvidia/AceInstruct"):
        model_kwargs.setdefault("device_map", "auto")
        if "torch_dtype" not in model_kwargs and quant_config is None:
            model_kwargs["torch_dtype"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        # Newer Qwen/DeepSeek families sometimes require remote code for custom tokenizers
        tokenizer_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True
        model_kwargs.setdefault("device_map", "auto")
        if "torch_dtype" not in model_kwargs and quant_config is None:
            model_kwargs["torch_dtype"] = "auto"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def _generate_from_messages(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    log_progress("_generate_from_messages: generating model output")
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    else:
        text = "\n".join(message.get("content", "") for message in messages)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95, top_k=20)
    prompt_length = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][prompt_length:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _separate_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Split ``text`` into (reply, thinking) parts when think tags are present."""
    log_progress("_separate_thinking: separating reply and thinking content")
    if not text:
        return "", None

    thinking_content: Optional[str] = None

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        text = (text[: match.start()] + text[match.end():]).strip()
        text = text.replace("<think>", "")
        return text.strip(), thinking_content

    if "</think>" in text:
        prefix, suffix = text.split("</think>", 1)
        thinking_content = prefix.replace("<think>", "").strip()
        return suffix.strip(), thinking_content
    print("no thinking tags found") 
    return text.strip(), None


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    observation: str,
    job: str,
    *,
    description_map: Optional[Dict[str, str]] = None,
    max_new_tokens: int = 5000,
) -> Tuple[str, Optional[str], PreTrainedModel]:
    """Feed the observation to the model and return ``(reply, thinking, model)``.

    A few model families emit internal chain-of-thought wrapped in ``</think>``
    tags; if present we strip it out and expose it via the second return value
    so the caller can decide whether to persist it.
    """
    log_progress(f"generate_response: generating response for job '{job}'")
    set_seed(42)
    messages = build_prompt(job, observation, description_map=description_map)
    raw_output = _generate_from_messages(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )
    reply, thinking_content = _separate_thinking(raw_output)
    return reply, thinking_content, model


def unload(model: Optional[PreTrainedModel]):
    """Free a model's memory footprint (CPU + GPU) when we are done."""
    log_progress("unload: releasing model resources")
    if model is None:
        return
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = None
    del model
    if dev and "cuda" in str(dev):
        # Release VRAM occupied by the model weights/kv cache
        torch.cuda.empty_cache()
    # Also nudge the Python GC for CPU tensors
    gc.collect()


# ---------------------------------------------------------------------------
# Runtime analysis helpers (prediction extraction, metrics, and I/O)
# ---------------------------------------------------------------------------


def build_variant_descriptions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    mode: str,
    iterations: Sequence[str],
    *,
    base_descriptions: Optional[Mapping[str, str]] = None,
    num_iterations: Optional[int] = None,
    max_new_tokens: int = 5000,
) -> Dict[str, Dict[str, str]]:
    """Generate per-iteration description variants using the requested ``mode``."""
    log_progress(f"build_variant_descriptions: starting {mode} variant generation")

    if mode not in {"constructive", "destructive", "vgdl"}:
        raise ValueError(f"Unknown variant generation mode: {mode}")

    description_source: Mapping[str, str] = (
        base_descriptions if base_descriptions is not None else descriptions
    )

    ordered_iterations = [str(it) for it in iterations]
    if num_iterations is not None:
        ordered_iterations = ordered_iterations[:num_iterations]

    variants: Dict[str, Dict[str, str]] = {}
    # Deterministic but distinct seed per iteration/game/mode/model combo
    seed_base = sum(ord(ch) for ch in f"{model_name}:{mode}")
    games = list(description_source.items())

    vgdl_cache: Dict[str, str] = {}

    for iter_idx, iteration in enumerate(ordered_iterations):
        log_progress(f"build_variant_descriptions: processing iteration {iteration}")
        per_iteration: Dict[str, str] = {}
        for game_idx, (game, original_description) in enumerate(games):
            if mode == "constructive":
                messages = build_prompt_constructive_description(game, original_description)
            elif mode == "destructive":
                messages = build_prompt_destructive_description(game, original_description)
            else:
                if game not in vgdl_cache:
                    vgdl_cache[game] = load_vgdl_source(game)
                messages = build_prompt_vgdl_description(game, vgdl_cache[game])

            # set_seed(seed_base + iter_idx * 997 + game_idx)
            rewritten = _generate_from_messages(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
            )
            rewritten = _separate_thinking(rewritten)[0]
            per_iteration[game] = rewritten

        variants[iteration] = per_iteration

    return variants

def extract_game_name(text: Optional[str]) -> str:
    """Extract a predicted game name from arbitrary model text.

    Returns one of:
      - exact game name from VALID_GAMES if exactly one is found
      - "multiple" if more than one different game name is found
      - "none" if none are found

    Rules mirror the notebook logic:
      - case-insensitive substring matching
      - special-case: any substring matching r"alien\w*" counts as "aliens"
    """
    log_progress("extract_game_name: attempting to extract game name")
    if not text or not isinstance(text, str):
        return "none"

    txt = text.lower()
    found = set()

    for game in VALID_GAMES:
        if game == "aliens":
            # Treat any word variant like "alien" or "aliens" as the same class
            if re.search(r"alien\w*", txt, flags=re.IGNORECASE):
                found.add("aliens")
        else:
            if re.search(re.escape(game), txt, flags=re.IGNORECASE):
                found.add(game)

    if len(found) == 1:
        return next(iter(found))
    elif len(found) > 1:
        # Notify callers that the output was ambiguous so they can inspect it
        warnings.warn("Multiple game names detected in output.", UserWarning)
        return "multiple"
    else:
        warnings.warn("No valid game name found in output.", UserWarning)
        return "none"


def is_classification_job(job: str) -> bool:
    """Return ``True`` if ``job`` expects a single game-name prediction."""
    log_progress(f"is_classification_job: evaluating job '{job}'")
    return job in {"wo_actions", "w_actions", "w_description"}


class OnlineMetrics:
    """Lightweight streaming metrics collector for classification jobs.

    The object keeps a running confusion matrix, overall accuracy, and per-label
    tallies so we can report metrics before persisting them.  It is intentionally
    minimal so it can be reused in notebooks where dependencies should stay
    small.
    """

    def __init__(self, labels: List[str]):
        log_progress("OnlineMetrics.__init__: initialising metrics tracker")
        self.labels: List[str] = list(labels)
        # confusion[true][pred] -> count
        self.confusion: Dict[str, Dict[str, int]] = {t: {} for t in self.labels}
        self.correct: int = 0
        self.attempts: int = 0
        self.per_label_attempts: Dict[str, int] = {t: 0 for t in self.labels}
        self.per_label_correct: Dict[str, int] = {t: 0 for t in self.labels}

    def record(self, true_label: str, prediction: str) -> None:
        log_progress(f"OnlineMetrics.record: true={true_label}, prediction={prediction}")
        if true_label not in self.confusion:
            # ignore unseen labels gracefully
            self.confusion[true_label] = {}
            self.per_label_attempts.setdefault(true_label, 0)
            self.per_label_correct.setdefault(true_label, 0)
        # Update the confusion matrix for this (true, predicted) pair
        self.confusion[true_label][prediction] = self.confusion[true_label].get(prediction, 0) + 1
        # Maintain overall counters for quick summaries
        self.attempts += 1
        self.per_label_attempts[true_label] = self.per_label_attempts.get(true_label, 0) + 1
        if prediction == true_label:
            self.correct += 1
            self.per_label_correct[true_label] = self.per_label_correct.get(true_label, 0) + 1

    def summary(self) -> Dict[str, object]:
        log_progress("OnlineMetrics.summary: computing summary statistics")
        # Compute macro stats on the fly to avoid storing redundant values
        overall_acc = (self.correct / self.attempts) if self.attempts else 0.0
        per_label = {}
        for label in self.per_label_attempts.keys():
            attempts = self.per_label_attempts.get(label, 0)
            correct = self.per_label_correct.get(label, 0)
            acc = (correct / attempts) if attempts else 0.0
            per_label[label] = {"correct": correct, "attempts": attempts, "accuracy": acc}
        return {
            "overall": {"correct": self.correct, "attempts": self.attempts, "accuracy": overall_acc},
            "per_label": per_label,
            "confusion": self.confusion,
        }


def save_metrics_json(metrics: Dict[str, object], path: Path) -> None:
    """Persist ``metrics`` to ``path`` as indented JSON."""
    log_progress(f"save_metrics_json: writing metrics to {path}")
    # Ensure the parent directories exist before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def save_predictions_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """Write a simple CSV of prediction rows.

    Each row should include: subset, iteration, game, model, reply, prediction, correct (bool)
    """
    log_progress(f"save_predictions_csv: writing predictions to {path}")
    # We intentionally use UTF-8 so replies/thoughts preserve punctuation
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset",
        "iteration",
        "game",
        "model",
        "reply",
        "thought",
        "prediction",
        "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # Only write the fields we care about, ignore any extras in the row dict
            writer.writerow({k: r.get(k, "") for k in fieldnames})
