import plistlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clr_codec


PALETTES = ["AWS 2021", "Apple", "Colors of Japan", "Open Color"]


def _hex_rgb(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _quantize(rgba):
    return tuple(round(component * 255) for component in rgba[:3])


def _hex_lines(stem):
    path = ROOT / "HEX" / f"{stem}.txt"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_keyed(path, items, raw_components=False):
    """Write a keyed NSColorList plist fixture using the NSKeys/NSColors schema."""

    count = len(items)
    array_class = 2
    color_class = 4 + 2 * count
    names_start = 4
    colors_start = 4 + count

    def array(refs):
        return {
            "$class": plistlib.UID(array_class),
            "NS.objects": [plistlib.UID(index) for index in refs],
        }

    objects = [
        "$null",
        array(range(names_start, names_start + count)),
        {
            "$classname": "NSMutableArray",
            "$classes": ["NSMutableArray", "NSArray", "NSObject"],
        },
        array(range(colors_start, colors_start + count)),
    ]
    objects.extend(name for name, _ in items)
    for _, (red, green, blue, alpha) in items:
        if raw_components:
            value = struct.pack("<4f", red, green, blue, alpha)
        else:
            components = f"{red:.9g} {green:.9g} {blue:.9g}"
            if alpha != 1:
                components += f" {alpha:.9g}"
            value = (components + "\0").encode("ascii")
        objects.append({
            "$class": plistlib.UID(color_class),
            "NSColorSpace": 1,
            "NSRGB": value,
        })
    objects.append({"$classname": "NSColor", "$classes": ["NSColor", "NSObject"]})

    with path.open("wb") as file:
        plistlib.dump(
            {
                "$version": 100000,
                "$archiver": "NSKeyedArchiver",
                "$top": {
                    "NSKeys": plistlib.UID(1),
                    "NSColors": plistlib.UID(3),
                },
                "$objects": objects,
            },
            file,
            fmt=plistlib.FMT_BINARY,
        )


class RepoClrTests(unittest.TestCase):
    def test_repo_clr_files_match_hex_sources(self):
        for stem in PALETTES:
            with self.subTest(palette=stem):
                items = clr_codec.read_clr(ROOT / "CLR" / f"{stem}.clr")
                lines = _hex_lines(stem)
                self.assertEqual(len(items), len(lines))
                for (name, rgba), line in zip(items, lines):
                    expected_hex, expected_name = line.split(maxsplit=1)
                    self.assertEqual(name, expected_name)
                    self.assertEqual(_quantize(rgba), _hex_rgb(expected_hex))
                    self.assertAlmostEqual(rgba[3], 1.0)

    def test_repo_clr_roundtrip(self):
        for stem in PALETTES:
            with self.subTest(palette=stem):
                items = clr_codec.read_clr(ROOT / "CLR" / f"{stem}.clr")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "roundtrip.clr"
                    clr_codec.write_clr(path, items)
                    restored = clr_codec.read_clr(path)
                self.assertEqual([name for name, _ in items], [name for name, _ in restored])
                self.assertEqual(
                    [_quantize(rgba) for _, rgba in items],
                    [_quantize(rgba) for _, rgba in restored],
                )


class WriterTests(unittest.TestCase):
    def test_empty_palette(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.clr"
            clr_codec.write_clr(path, [])
            self.assertEqual(path.stat().st_size, 26)
            self.assertEqual(clr_codec.read_clr(path), [])

    def test_unicode_alpha_and_order(self):
        items = [
            ("Zebra", (1.0, 0.0, 0.0, 1.0)),
            ("Grün 色", (0.2, 0.4, 0.6, 0.5)),
            ("最後", (0.0, 1.0, 0.0, 0.0)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unicode.clr"
            clr_codec.write_clr(path, items)
            restored = clr_codec.read_clr(path)

        self.assertEqual([name for name, _ in restored], [name for name, _ in items])
        for (_, expected), (_, actual) in zip(items, restored):
            for expected_component, actual_component in zip(expected, actual):
                self.assertAlmostEqual(expected_component, actual_component, places=6)

    def test_three_components_default_to_opaque(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.clr"
            clr_codec.write_clr(path, [("Red", (1.0, 0.0, 0.0))])
            restored = clr_codec.read_clr(path)
        self.assertEqual(restored[0][1], (1.0, 0.0, 0.0, 1.0))

    def test_rejects_invalid_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.clr"
            with self.assertRaises(ValueError):
                clr_codec.write_clr(path, [("Bad", (1.2, 0.0, 0.0))])
            with self.assertRaises(ValueError):
                clr_codec.write_clr(path, [("Bad", (0.0, 0.0))])


class AlternativeFormatTests(unittest.TestCase):
    def test_reads_keyed_plist(self):
        items = [("Red", (1.0, 0.0, 0.0, 1.0)), ("Grün 色", (0.2, 0.4, 0.6, 0.5))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keyed.clr"
            _write_keyed(path, items)
            restored = clr_codec.read_clr(path)
        self.assertEqual([name for name, _ in restored], [name for name, _ in items])
        self.assertEqual(restored[1][1], (0.2, 0.4, 0.6, 0.5))

    def test_reads_keyed_raw_float_components(self):
        items = [("Raw", (0.1, 0.2, 0.3, 0.4))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.clr"
            _write_keyed(path, items, raw_components=True)
            restored = clr_codec.read_clr(path)
        self.assertEqual(restored[0][0], "Raw")
        for expected, actual in zip(items[0][1], restored[0][1]):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_reads_keyed_named_color_fallback(self):
        objects = [
            "$null",
            {
                "$class": plistlib.UID(2),
                "NS.objects": [plistlib.UID(4)],
            },
            {
                "$classname": "NSMutableArray",
                "$classes": ["NSMutableArray", "NSArray", "NSObject"],
            },
            {
                "$class": plistlib.UID(2),
                "NS.objects": [plistlib.UID(5)],
            },
            "labelColor",
            {
                "$class": plistlib.UID(6),
                "NSColorSpace": 6,
                "NSColor": plistlib.UID(7),
            },
            {"$classname": "NSColor", "$classes": ["NSColor", "NSObject"]},
            {
                "$class": plistlib.UID(6),
                "NSColorSpace": 1,
                "NSWhite": b"0.5\0",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "named.clr"
            with path.open("wb") as file:
                plistlib.dump(
                    {
                        "$version": 100000,
                        "$archiver": "NSKeyedArchiver",
                        "$top": {
                            "NSKeys": plistlib.UID(1),
                            "NSColors": plistlib.UID(3),
                        },
                        "$objects": objects,
                    },
                    file,
                    fmt=plistlib.FMT_BINARY,
                )
            restored = clr_codec.read_clr(path)
        self.assertEqual(restored, [("labelColor", (0.5, 0.5, 0.5, 1.0))])

    def test_keyed_plist_requires_nscolorlist_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-a-clr.clr"
            with path.open("wb") as file:
                plistlib.dump(
                    {"$top": {"root": plistlib.UID(1)}, "$objects": ["$null", "x"]},
                    file,
                    fmt=plistlib.FMT_BINARY,
                )
            with self.assertRaises(clr_codec.ClrError):
                clr_codec.read_clr(path)

    def test_reads_ascii(self):
        text = "3\n0 1.0 0.0 0.0 1.0 Red\n0 0.2 0.4 0.6 0.5 Grün 色\n0 0 1 0 Green\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ascii.clr"
            path.write_text(text, encoding="utf-8")
            restored = clr_codec.read_clr(path)
        self.assertEqual([name for name, _ in restored], ["Red", "Grün 色", "Green"])
        self.assertEqual(restored[1][1], (0.2, 0.4, 0.6, 0.5))
        self.assertEqual(restored[2][1], (0.0, 1.0, 0.0, 1.0))

    def test_rejects_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.clr"
            path.write_text("this is not a color list\n", encoding="utf-8")
            with self.assertRaises(clr_codec.ClrError):
                clr_codec.read_clr(path)


if __name__ == "__main__":
    unittest.main()
