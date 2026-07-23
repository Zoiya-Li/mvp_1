"""Curated portrait-shoot catalog built on the existing template library.

The public catalog is intentionally smaller than the raw prompt library. One
catalog entry represents one deliverable six-frame shoot, never a broad style
bucket containing unrelated people, wardrobes, and locations.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from . import storage
from .portrait_storage import (
    list_themes,
    mark_unlisted_themes_legacy,
    upsert_theme,
)


_FRAME_BLUEPRINTS: list[dict[str, str]] = [
    {
        "shot_id": "closeup",
        "label": "开场近景肖像",
        "framing": "natural chest-up portrait with breathing room above the head",
        "pose": "subtle three-quarter turn, relaxed jaw, gaze close to the camera",
        "lens": "50mm to 70mm lens at f/2.8 to f/4",
        "narrative": "the first quiet encounter",
    },
    {
        "shot_id": "half_body",
        "label": "半身肖像",
        "framing": "complete head-to-waist portrait with the garment silhouette readable",
        "pose": "relaxed shoulders, slight turn, arms resting naturally outside the crop",
        "lens": "50mm lens at f/3.2 to f/4",
        "narrative": "the person settles into the space",
    },
    {
        "shot_id": "environmental",
        "label": "环境人像",
        "framing": "head-to-upper-thigh environmental portrait with the setting occupying at least half the frame",
        "pose": "standing off-center with natural weight shift and an unperformed expression",
        "lens": "35mm to 50mm lens at f/4",
        "narrative": "the room and the person belong to the same moment",
    },
    {
        "shot_id": "seated",
        "label": "坐姿肖像",
        "framing": "complete head-to-waist seated portrait with chair and posture clearly visible",
        "pose": "open shoulders and a slight turn; hands only when complete and anatomically natural",
        "lens": "50mm lens at f/3.2 to f/4",
        "narrative": "a pause inside the session",
    },
    {
        "shot_id": "profile",
        "label": "侧转肖像",
        "framing": "chest-up turned portrait with asymmetric shoulders and negative space",
        "pose": "face and shoulders turned 25 to 40 degrees, gaze just beyond the frame",
        "lens": "50mm to 70mm lens at f/3.2 to f/4",
        "narrative": "attention briefly moves beyond the camera",
    },
    {
        "shot_id": "candid",
        "label": "收尾抓拍",
        "framing": "chest-up to half-body candid portrait with enough surroundings to identify the same place",
        "pose": "a quiet in-between movement or breath, expression unperformed",
        "lens": "50mm documentary lens at f/4",
        "narrative": "the session ends on an unguarded frame",
    },
]


_SHOOT_SERIES: list[dict[str, Any]] = [
    {
        "key": "white-cotton-daylight",
        "engine_style_key": "jk_portrait",
        "template_id": "jp_f_fresh",
        "title": "白棉布与晴日",
        "title_en": "White Cotton, Open Shade",
        "tagline": "哑光白棉、柔和日光，以及一间安静的海边小屋。",
        "category": "自然光",
        "featured": True,
        "use_cases": ["个人写真", "生日纪念", "社交头像"],
        "wardrobe": "one matte white cotton dress with visible weave, natural folds, and no jewelry change",
        "lighting": "soft open-shade daylight with ordinary contrast and no glow effect",
        "style_prompt": "Unretouched natural-light editorial photography from one late-spring session. Keep matte white cotton, restrained whites and soft greens, realistic fabric folds, ordinary skin color variation, subtle camera grain, and the calm color response of a 50mm color-negative photograph. The result is observed, not posed or beautified.",
        "environments": [
            "beside an open window in a lived-in whitewashed seaside room, with the timber frame and curtain readable",
            "inside the same room near a plain wooden table and a ceramic water glass",
            "in the adjoining doorway where the room opens onto a shaded veranda",
            "seated on a simple wooden chair on that same veranda",
            "turned beside the moving cotton curtain at the original window",
            "taking one slow step along the veranda with the same house and pale sky behind",
        ],
    },
    {
        "key": "window-light-silk",
        "engine_style_key": "jk_portrait",
        "template_id": "kr_f_elegant",
        "title": "窗光与丝绸",
        "title_en": "Window Light & Silk",
        "tagline": "奶油色丝绸、暖调石材，以及有方向的窗光。",
        "category": "静谧室内",
        "featured": True,
        "use_cases": ["个人写真", "周年纪念", "社交头像"],
        "wardrobe": "one cream silk blouse with restrained sheen and dark tailored trousers, unchanged across the set",
        "lighting": "directional north-window daylight with soft but visible facial shadow",
        "style_prompt": "A restrained contemporary editorial session in one real apartment interior. Preserve cream silk texture, warm-gray stone, dark timber, quiet neutral color, realistic pores and under-eye structure, and moderate depth of field. Makeup remains minimal and the camera response stays crisp rather than luminous or dewy.",
        "environments": [
            "at a tall apartment window with dark timber trim and a real street layer beyond",
            "one step back from the same window beside a warm-gray stone wall",
            "in the connected living room where the window, low sofa, and timber floor remain readable",
            "seated at the end of that low sofa beside one practical floor lamp",
            "turned toward the original window with the stone wall receding behind",
            "adjusting the blouse cuff beside the same sofa in an unposed closing moment",
        ],
    },
    {
        "key": "old-shanghai-evening",
        "engine_style_key": "chinese_style",
        "template_id": "gf_f_qipao",
        "title": "旧上海黄昏",
        "title_en": "Old Shanghai Evening",
        "tagline": "墨绿旗袍、琥珀灯光，以及被雨浸深的砖墙。",
        "category": "东方旧影",
        "featured": True,
        "use_cases": ["旗袍写真", "个人写真", "周年纪念"],
        "wardrobe": "one dark green silk qipao with subtle woven pattern and restrained vintage hair styling",
        "lighting": "mixed dusk window light and warm tungsten practicals with believable falloff",
        "style_prompt": "A grounded Republican-era inspired portrait session photographed in one preserved lane-house interior and its adjoining brick alley at dusk. Keep dark green silk, amber practical light, rain-dark brick, natural skin texture, restrained film grain, and physically coherent period details. Avoid costume-drama polish; it should feel like a real location portrait.",
        "environments": [
            "beside a lane-house window with dark wood, lace curtain, and dusk-blue alley light",
            "inside the same room near a small round table and one warm shaded lamp",
            "in the connected brick doorway with both the room and narrow alley readable",
            "seated on a bentwood chair beside the same table and lamp",
            "turned in profile at the original window as the alley light falls across the face",
            "walking slowly just outside the doorway on the damp brick lane",
        ],
    },
    {
        "key": "lamp-lit-room",
        "engine_style_key": "cinematic",
        "template_id": "film_f_cinematic",
        "title": "一盏灯的房间",
        "title_en": "The Lamp-Lit Room",
        "tagline": "一间暗室，一盏台灯，黄昏后的六个镜头。",
        "category": "电影感",
        "featured": False,
        "use_cases": ["电影感写真", "个人写真", "社交头像"],
        "wardrobe": "one charcoal knit top with visible texture and a simple dark skirt or trousers",
        "lighting": "one warm table lamp balanced against faint blue dusk from the window",
        "style_prompt": "An intimate unretouched cinema-lens portrait session in one ordinary apartment room after dusk. Keep warm amber practical light, faint cool window fill, deep but detailed shadows, visible bookshelf and curtain texture, realistic skin, and restrained 35mm grain. The images should feel witnessed, not staged as a movie poster.",
        "environments": [
            "near a small table lamp with a bookshelf and curtain legible in the same dark room",
            "standing between the lamp table and the dusk window in that room",
            "farther back in the room with the lamp, shelf, chair, and window defining the space",
            "seated in the room's worn armchair beside the same lamp",
            "turned toward the dusk window while warm lamp light remains on the far cheek",
            "closing a book beside the lamp in a quiet unperformed moment",
        ],
    },
    {
        "key": "forest-clearing",
        "engine_style_key": "creative",
        "template_id": "cr_f_forest",
        "title": "林间空地",
        "title_en": "The Forest Clearing",
        "tagline": "白棉布、真实枝叶，以及穿过树隙的午后阳光。",
        "category": "户外自然",
        "featured": False,
        "use_cases": ["户外写真", "个人写真", "生日纪念"],
        "wardrobe": "one simple white cotton midi dress with natural wrinkles; no hat or bouquet added unless already present",
        "lighting": "broken afternoon sunlight filtered through real leaves, with controlled highlights",
        "style_prompt": "A natural outdoor editorial session made in one forest clearing and the footpath beside it. Keep real leaf detail, damp earth, restrained sage green, white cotton texture, uneven filtered sunlight, ordinary skin texture, flyaway hair, and moderate depth of field. No fantasy haze, glowing particles, or storybook treatment.",
        "environments": [
            "at the shaded edge of a real forest clearing with bark and fern detail readable",
            "two steps into the same clearing beside a low mossy trunk",
            "on the adjoining narrow footpath with the original clearing visible behind",
            "seated on the same low fallen trunk with both feet and posture naturally supported",
            "turned toward a shaft of light at the clearing edge",
            "walking back onto the footpath as leaves move in a light breeze",
        ],
    },
    {
        "key": "linen-open-shade",
        "engine_style_key": "jk_portrait",
        "template_id": "jp_m_fresh",
        "title": "亚麻与阴凉处",
        "title_en": "Linen in Open Shade",
        "tagline": "白色亚麻、街区日光，没有影棚式的过度精修。",
        "category": "自然光",
        "featured": False,
        "use_cases": ["个人写真", "社交头像", "生日纪念"],
        "wardrobe": "one softly worn white linen shirt with visible weave and natural creases",
        "lighting": "bright open shade with gentle direction and ordinary local contrast",
        "style_prompt": "An observational neighborhood portrait session in open shade. Keep softly worn white linen, muted green and concrete tones, real fabric creases, natural hair, skin texture, restrained color-negative grain, and 50mm perspective. Expressions should occur between poses rather than read as fashion modeling.",
        "environments": [
            "under the awning of a quiet neighborhood cafe with its window frame and street reflection readable",
            "beside the same cafe window and a plain outdoor table",
            "at the connected sidewalk edge with storefront and paving depth visible",
            "seated on a simple cafe chair under the same awning",
            "turned toward the street while the cafe window remains behind",
            "taking a slow step past the storefront in the same open shade",
        ],
    },
    {
        "key": "quiet-grey-room",
        "engine_style_key": "jk_portrait",
        "template_id": "kr_m_minimal",
        "title": "安静的灰色房间",
        "title_en": "The Quiet Grey Room",
        "tagline": "黑色针织、暖灰石材，一间克制安静的房间。",
        "category": "静谧室内",
        "featured": False,
        "use_cases": ["个人写真", "社交头像", "杂志感肖像"],
        "wardrobe": "one matte black knit turtleneck with visible fiber texture and dark tailored trousers",
        "lighting": "directional side-window daylight with clean shadow shape and no glow",
        "style_prompt": "A disciplined editorial portrait session in one real modern room. Keep matte black knit, warm-gray stone, dark timber, restrained neutral color, subtle facial asymmetry, realistic skin, and medium-format clarity without cosmetic retouching. The room remains physically readable and never becomes a seamless studio backdrop.",
        "environments": [
            "beside a tall side window set into a warm-gray stone wall",
            "one step into the same room where a timber bench and stone floor are visible",
            "in the connected passage with the original window and room depth behind",
            "seated on the same timber bench with posture and room geometry clear",
            "turned toward the side window with asymmetric shoulders",
            "looking down briefly while adjusting one sleeve beside the bench",
        ],
    },
    {
        "key": "hong-kong-supper",
        "engine_style_key": "creative",
        "template_id": "cr_m_retro",
        "title": "夜宵摊之后",
        "title_en": "After the Late Supper",
        "tagline": "卷起的白衬衫、钨丝灯下的蒸汽，以及潮湿街色。",
        "category": "城市夜景",
        "featured": False,
        "use_cases": ["夜景写真", "个人写真", "社交头像"],
        "wardrobe": "one slightly oversized white button-up shirt with rolled sleeves and natural wear",
        "lighting": "mixed tungsten stall light and dim green-orange street practicals",
        "style_prompt": "A grounded late-evening street portrait session around one working food stall and the lane beside it. Keep rolled white linen sleeves, tungsten steam, wet pavement, imperfect green-orange practical color, visible grain, real skin texture, and handheld 50mm perspective. No light leaks, nostalgia overlay, imitation film damage, or romantic haze.",
        "environments": [
            "at the edge of a working street-food stall with steel counter, steam, and menu details readable",
            "beside the same counter under one tungsten work light",
            "in the narrow lane immediately outside with the stall still visible behind",
            "seated on a simple folding stool at the stall's side table",
            "turned toward passing street light while the counter remains in the background",
            "leaving the stall along the same wet lane in an unposed final frame",
        ],
    },
    {
        "key": "neon-after-rain",
        "engine_style_key": "cinematic",
        "template_id": "film_m_cyber",
        "title": "雨后的霓虹",
        "title_en": "Neon After Rain",
        "tagline": "深色层次、真实店铺灯光，以及雨后发亮的路面。",
        "category": "城市夜景",
        "featured": False,
        "use_cases": ["夜景写真", "个人写真", "杂志感肖像"],
        "wardrobe": "one dark matte hooded jacket worn open over a plain black shirt",
        "lighting": "mixed cyan shop light and restrained magenta signage reflected by wet pavement",
        "style_prompt": "A real night street editorial session after rain, using practical shop and sign light rather than science-fiction effects. Keep dark matte layers, wet pavement texture, restrained cyan and magenta spill, realistic shadow detail, visible skin texture, and moderate 50mm depth of field. No lens flare, anime treatment, futuristic props, or synthetic neon fog.",
        "environments": [
            "under a real shop awning with wet pavement and two practical colored lights",
            "beside the same shop window with reflected street lettering readable",
            "at the mouth of the adjoining lane with the original awning behind",
            "seated on a dry concrete ledge under that awning",
            "turned toward traffic light reflected in the shop glass",
            "walking into the adjoining wet lane while the same shop light recedes",
        ],
    },
    {
        "key": "hard-light-black",
        "engine_style_key": "fashion",
        "template_id": "fz_f_editorial",
        "title": "黑衣与硬光",
        "title_en": "Black Cloth, Hard Light",
        "tagline": "利落黑色剪裁，置于真实工作的日光影棚。",
        "category": "时装影棚",
        "featured": False,
        "use_cases": ["时装写真", "个人写真", "社交头像"],
        "wardrobe": "one structured oversized black blazer over a simple opaque black base garment",
        "lighting": "one hard daylight beam shaped by the studio window, with honest falloff and shadow",
        "style_prompt": "A sharp fashion editorial made in one working daylight studio, with the room's stands, timber floor, paper roll edge, and window physically readable. Keep structured black tailoring, cool neutral color, hard directional light, realistic skin pores, subtle fabric lint and creases, and medium-format detail without beauty retouching.",
        "environments": [
            "near the working studio window with one stand and the edge of a paper roll visible",
            "against the same studio's textured plaster wall crossed by a hard daylight beam",
            "farther back on the timber floor with window, stand, and wall defining the room",
            "seated on a plain studio apple box inside the same light path",
            "turned at the edge of the hard beam with the working studio receding behind",
            "stepping out of the light while adjusting the blazer hem in the same room",
        ],
    },
]


def _catalog_image_url(template_image: str) -> str:
    stem = Path(template_image).stem
    return f"/api/v2/catalog-images/{stem}.jpg?v=3"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _shots_for(series: dict[str, Any]) -> list[dict[str, str]]:
    environments = series["environments"]
    if len(environments) != len(_FRAME_BLUEPRINTS):
        raise ValueError(f"{series['key']} must define six environments")
    return [
        {
            **frame,
            "environment": environments[index],
            "lighting": series["lighting"],
            "wardrobe": series["wardrobe"],
            "style_prompt": series["style_prompt"],
        }
        for index, frame in enumerate(_FRAME_BLUEPRINTS)
    ]


def seed_theme_catalog() -> None:
    prompts_path = Path(__file__).resolve().parent.parent / "prompts.json"
    prompt_library = json.loads(prompts_path.read_text(encoding="utf-8"))
    styles = prompt_library.get("styles", {})
    now = storage.utcnow()
    active_source_keys: set[str] = set()

    for index, series in enumerate(_SHOOT_SERIES):
        style_key = series["engine_style_key"]
        style = styles.get(style_key) or {}
        template = next(
            (
                item for item in style.get("templates", [])
                if item.get("id") == series["template_id"]
            ),
            None,
        )
        if not template or not template.get("template_image"):
            raise RuntimeError(
                f"Catalog shoot {series['key']} references a missing template"
            )

        source_key = f"shoot_{series['key'].replace('-', '_')}"
        active_source_keys.add(source_key)
        stable = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"flashshot-shoot:{series['key']}",
        ).hex
        shots = _shots_for(series)
        preview = _catalog_image_url(template["template_image"])
        upsert_theme({
            "theme_id": f"thm_{stable}",
            "slug": _slug(series["title_en"]),
            "title": series["title"],
            "title_en": series["title_en"],
            "tagline": series["tagline"],
            "category": series["category"],
            "cover_image": preview,
            "preview_images": [preview],
            "use_cases": series["use_cases"],
            "source_style_key": source_key,
            "featured": series["featured"],
            "sort_order": index,
            "blueprint": {
                "engine_style_key": style_key,
                "template_id": template["id"],
                "presentation": template.get("gender", "unspecified"),
                "set_size": 6,
                "reference_min": 4,
                "reference_max": 6,
                "shots": shots,
                "templates": [{
                    "template_id": template["id"],
                    "label": template.get("label"),
                    "gender": template.get("gender", "unspecified"),
                    "template_image": template["template_image"],
                    "shots": shots,
                }],
                "preview_integrity": "single_direction_study",
                "quality_policy": "portrait_series_v2",
            },
        }, now)

    mark_unlisted_themes_legacy(active_source_keys, now)


def ensure_theme_catalog() -> None:
    seed_theme_catalog()
    if not list_themes():
        raise RuntimeError("Portrait theme catalog could not be initialized")
