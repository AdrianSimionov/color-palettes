"""Portable reader/writer for macOS color list (``.clr``) files.

The module understands the three CLR encodings found in the wild:

* legacy NSArchiver typedstream files (read with :mod:`pytypedstream`)
* keyed NSKeyedArchiver binary plists (read with :mod:`plistlib`)
* the legacy ASCII color-list representation

Writing uses the legacy typedstream representation, which is what Apple's
``NSColorList`` itself produces.  It keeps the output readable by older macOS
versions and does not require AppKit or PyObjC.

Public API::

    items = read_clr("Colors.clr")
    # [(name, (red, green, blue, alpha)), ...]

    write_clr("Colors.clr", [
        ("Red", (1.0, 0.0, 0.0, 1.0)),
    ])

Components are floats in the 0.0..1.0 range, matching ``NSColor``.
"""

import math
import plistlib
import re
import struct

__all__ = ["ClrError", "read_clr", "write_clr"]


class ClrError(ValueError):
    """Raised when a CLR file cannot be decoded or encoded."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_clr(path):
    """Read *path* and return ``[(name, (r, g, b, a)), ...]``.

    Raises :class:`ClrError` if the file is not a recognized CLR format or
    cannot be decoded.
    """

    with open(path, "rb") as fp:
        data = fp.read()

    if data.startswith(b"bplist00"):
        return _read_keyed(data)
    if b"streamtyped" in data[:32] or b"typedstream" in data[:32]:
        return _read_typedstream(data)
    return _read_ascii(data)


def _read_typedstream(data):
    try:
        from typedstream import Unarchiver
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ClrError(
            "pytypedstream is required to read legacy CLR files. "
            "Install it with 'pip install pytypedstream'."
        ) from exc

    try:
        groups = Unarchiver.from_data(data).decode_all()
    except Exception as exc:
        raise ClrError(f"Unable to decode typedstream CLR: {exc}") from exc

    items = []
    for group in groups:
        values = getattr(group, "values", ())
        if len(values) != 2:
            continue
        color, name = values
        if type(color).__name__ != "NSColor":
            continue
        items.append((_string_value(name), _ns_color_to_rgba(color)))
    return items


def _ns_color_to_rgba(color, seen=None):
    """Convert a pytypedstream ``NSColor`` object to normalized RGBA."""

    if seen is None:
        seen = set()
    if id(color) in seen:
        raise ClrError("Cycle detected in archived NSColor objects")
    seen.add(id(color))

    kind = getattr(getattr(color, "kind", None), "name", None)
    value = getattr(color, "value", None)
    if value is None:
        raise ClrError(f"Unsupported NSColor object: {color!r}")

    if kind in ("CALIBRATED_RGBA", "DEVICE_RGBA"):
        return (value.red, value.green, value.blue, value.alpha)
    if kind in ("CALIBRATED_WA", "DEVICE_WA"):
        return (value.white, value.white, value.white, value.alpha)
    if kind == "DEVICE_CMYKA":
        red = (1.0 - value.cyan) * (1.0 - value.black)
        green = (1.0 - value.magenta) * (1.0 - value.black)
        blue = (1.0 - value.yellow) * (1.0 - value.black)
        return (red, green, blue, value.alpha)
    if kind == "NAMED":
        return _ns_color_to_rgba(value.color, seen)

    raise ClrError(f"Unsupported NSColor kind: {kind}")


def _read_keyed(data):
    try:
        archive = plistlib.loads(data)
    except Exception as exc:
        raise ClrError(f"Unable to decode keyed CLR plist: {exc}") from exc

    top = archive.get("$top")
    objects = archive.get("$objects")
    if not isinstance(top, dict) or not isinstance(objects, list):
        raise ClrError("Keyed CLR is missing the NSKeyedArchiver object table")

    if "NSKeys" not in top or "NSColors" not in top:
        raise ClrError(
            "Keyed CLR does not use the NSColorList schema "
            "(expected $top.NSKeys and $top.NSColors)"
        )

    names = _keyed_array(_resolve(top["NSKeys"], objects), objects)
    colors = _keyed_array(_resolve(top["NSColors"], objects), objects)
    if len(names) != len(colors):
        raise ClrError(
            f"Keyed CLR has {len(names)} names but {len(colors)} colors"
        )

    return [
        (_string_value(name), _keyed_color_to_rgba(color, objects))
        for name, color in zip(names, colors)
    ]


def _resolve(value, objects):
    """Resolve a plistlib UID reference to its object-table entry."""

    uid_type = getattr(plistlib, "UID", None)
    if uid_type is not None and isinstance(value, uid_type):
        try:
            return objects[value.data]
        except (IndexError, TypeError) as exc:
            raise ClrError(f"Invalid keyed CLR object reference: {value!r}") from exc
    return value


def _keyed_array(value, objects):
    value = _resolve(value, objects)
    if isinstance(value, dict) and "NS.objects" in value:
        return [_resolve(item, objects) for item in value["NS.objects"]]
    if isinstance(value, list):
        return [_resolve(item, objects) for item in value]
    raise ClrError("Keyed CLR NSKeys/NSColors entry is not an array")


def _keyed_color_to_rgba(color, objects, seen=None):
    color = _resolve(color, objects)
    if not isinstance(color, dict):
        raise ClrError(f"Invalid keyed CLR color object: {color!r}")

    if seen is None:
        seen = set()
    if id(color) in seen:
        raise ClrError("Cycle detected in keyed NSColor objects")
    seen.add(id(color))

    for key in ("NSRGB", "NSWhite", "NSComponents"):
        if key not in color:
            continue
        components = _parse_components(_resolve(color[key], objects))
        if key == "NSRGB":
            if len(components) == 3:
                return (*components, 1.0)
            if len(components) == 4:
                return tuple(components)
        elif key == "NSWhite":
            if len(components) == 1:
                return (components[0], components[0], components[0], 1.0)
            if len(components) == 2:
                white, alpha = components
                return (white, white, white, alpha)
        elif key == "NSComponents":
            if len(components) == 3:
                return (*components, 1.0)
            if len(components) == 4:
                return tuple(components)
            if len(components) == 5:
                cyan, magenta, yellow, black, alpha = components
                return (
                    (1.0 - cyan) * (1.0 - black),
                    (1.0 - magenta) * (1.0 - black),
                    (1.0 - yellow) * (1.0 - black),
                    alpha,
                )
        raise ClrError(
            f"Unsupported {key} component count: {len(components)}"
        )

    # Named/catalog colors carry the concrete NSColor as an NSColor fallback.
    if "NSColor" in color:
        return _keyed_color_to_rgba(color["NSColor"], objects, seen)

    raise ClrError("Keyed CLR color has no NSRGB, NSWhite, or NSComponents data")


def _parse_components(raw):
    """Parse numeric color components from an NSColor archive value."""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("ascii").strip("\x00 \t\r\n")
        except UnicodeDecodeError:
            text = None
        if text and re.fullmatch(r"[-+0-9eE.,\s]+", text):
            values = [part for part in re.split(r"[\s,]+", text) if part]
            try:
                return [float(part) for part in values]
            except ValueError:
                pass

        # Some keyed archives store a raw little-endian float array.
        if len(raw) in (12, 16):
            count = len(raw) // 4
            return list(struct.unpack(f"<{count}f", raw))
        raise ClrError(f"Unsupported color component data: {raw!r}")

    if isinstance(raw, str):
        values = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
        try:
            return [float(part) for part in values]
        except ValueError as exc:
            raise ClrError(f"Invalid color component data: {raw!r}") from exc

    raise ClrError(f"Unsupported color component type: {type(raw).__name__}")


def _read_ascii(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # The historical ASCII CLR format used MacRoman names.
        text = data.decode("mac_roman")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ClrError("Empty CLR file")

    match = re.fullmatch(r"\[?(\d+)(?:/colors)?\]?", lines[0])
    if not match:
        raise ClrError("Unsupported CLR format (not typedstream, keyed plist, or ASCII)")

    count = int(match.group(1))
    body = lines[1:]
    if len(body) < count:
        raise ClrError(f"ASCII CLR declares {count} colors but has {len(body)} lines")

    items = []
    for line in body[:count]:
        parts = line.split(None, 4)
        if len(parts) < 5:
            raise ClrError(f"Invalid ASCII CLR line: {line!r}")
        try:
            red, green, blue = (float(part) for part in parts[1:4])
        except ValueError as exc:
            raise ClrError(f"Invalid ASCII CLR line: {line!r}") from exc

        rest = parts[4]
        alpha = 1.0
        name = rest
        tokens = rest.split(None, 1)
        if tokens:
            try:
                alpha = float(tokens[0])
                name = tokens[1] if len(tokens) > 1 else ""
            except ValueError:
                pass
        items.append((name, (red, green, blue, alpha)))

    return items


def _string_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("NS.string", "value"):
            if key in value:
                return _string_value(value[key])
        return str(value)
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


_HEADER = bytes.fromhex("040b73747265616d747970656481e803")
_COUNT_PREFIX = bytes.fromhex("84016901" "84024069" "85")  # [i] 1, [@ i] nil
_FIRST_COLOR = bytes.fromhex(
    "84024040"
    "848484074e53436f6c6f7200"
    "8484084e534f626a65637400"
    "85"
    "84016301"
    "840466666666"
)
_FIRST_NAME = bytes.fromhex("848484084e53537472696e67019484012b")
_NEXT_COLOR = bytes.fromhex("948493970198")
_NEXT_NAME = bytes.fromhex("84969a")
_OBJECT_END = b"\x86"


def write_clr(path, items):
    """Write *items* to *path* as a legacy typedstream CLR.

    *items* is an iterable of ``(name, (r, g, b[, a]))`` pairs.  Components
    must be in the 0.0..1.0 range.  Names should be unique; ``NSColorList``
    uses them as keys.
    """

    prepared = []
    for name, components in items:
        components = tuple(float(component) for component in components)
        if len(components) == 3:
            components += (1.0,)
        if len(components) != 4:
            raise ValueError(
                f"Color {name!r} must have three or four components, "
                f"not {len(components)}"
            )
        for component in components:
            if not math.isfinite(component) or not 0.0 <= component <= 1.0:
                raise ValueError(
                    f"Color {name!r} has a component outside 0.0..1.0: "
                    f"{component!r}"
                )
        prepared.append((str(name), components))

    with open(path, "wb") as fp:
        fp.write(_encode_typedstream(prepared))


def _encode_typedstream(items):
    out = bytearray(_HEADER)
    out += _COUNT_PREFIX
    out += _encode_int(len(items))

    for index, (name, components) in enumerate(items):
        out += _FIRST_COLOR if index == 0 else _NEXT_COLOR
        for component in components:
            out += _encode_float(component)
        out += _OBJECT_END

        out += _FIRST_NAME if index == 0 else _NEXT_NAME
        out += _encode_string(name)
        out += _OBJECT_END

    return bytes(out)


def _encode_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"Cannot encode a negative integer: {value}")
    if value <= 0x7F:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\x81" + struct.pack("<H", value)
    if value <= 0xFFFFFFFF:
        return b"\x82" + struct.pack("<I", value)
    raise ValueError(f"Integer is too large for typedstream: {value}")


def _encode_float(value):
    if value == 0.0:
        return b"\x00"
    if value == 1.0:
        return b"\x01"
    return b"\x83" + struct.pack("<f", value)


def _encode_string(value):
    encoded = value.encode("utf-8")
    return _encode_int(len(encoded)) + encoded
