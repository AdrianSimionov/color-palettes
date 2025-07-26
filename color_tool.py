#!/usr/bin/python3

import argparse
import colorsys
import math
from typing import Tuple
import os
import json
import functools
import objc
import struct
import sys
import io
from Foundation import *
from AppKit import NSColorList, NSColor

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6 or not all(c in '0123456789ABCDEFabcdef' for c in hex_color):
        raise ValueError(f"Invalid hex color code: '{hex_color}'")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hsl_to_rgb(hsl_color):
    h, s, l = hsl_color
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h / 360, l / 100, s / 100))

def convert_to_hex(color):
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"

def convert_to_hsl(color):
    r_scaled, g_scaled, b_scaled = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r_scaled, g_scaled, b_scaled)
    h = int(h * 360)
    s = int(s * 100)
    l = int(l * 100)
    return f"HSL({h}, {s}%, {l}%)"

def convert_to_rgb(color):
    return f"RGB({color[0]}, {color[1]}, {color[2]})"

def format_color(color, color_format):
    if color_format == 'HSL':
        return convert_to_hsl(color)
    elif color_format == 'RGB':
        return convert_to_rgb(color)
    elif color_format == 'HEX':
        return convert_to_hex(color)
    else:
        raise ValueError(f"Unsupported color format: {color_format}")

def parse_input_color(color_str):
    color_str = color_str.strip()
    if color_str.startswith('#'):
        return hex_to_rgb(color_str)
    elif color_str.lower().startswith("rgb"):
        color_str = color_str[color_str.find("(")+1:color_str.find(")")].strip()
        return tuple(map(int, color_str.split(",")))
    elif color_str.lower().startswith("hsl"):
        color_str = color_str[color_str.find("(")+1:color_str.find(")")].strip()
        h, s, l = map(lambda x: float(x.strip('%')), color_str.split(","))
        return hsl_to_rgb((h, s, l))
    else:
        raise ValueError(f"Unknown color format: {color_str}")

def lum(r, g, b):
    return math.sqrt(.241 * r + .691 * g + .068 * b)

def step(r, g, b, repetitions=1):
    lum_value = math.sqrt(.241 * r + .691 * g + .068 * b)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h2 = int(h * repetitions)
    lum2 = int(lum_value * repetitions)
    v2 = int(v * repetitions)
    return (h2, lum2, v2)

def step_alternated(r, g, b, repetitions=1):
    lum_value = math.sqrt(.241 * r + .691 * g + .068 * b)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h2 = int(h * repetitions)
    lum2 = int(lum_value * repetitions)
    v2 = int(v * repetitions)
    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum2 = repetitions - lum2
    return (h2, lum2, v2)

class HillbertIndexCalculator:
    def calculate(self, rgb):
        x_coordinate = rgb[0]
        y_coordinate = rgb[1]
        z_coordinate = rgb[2]
        return self._get_int_from_hillbert_coordinates([x_coordinate, y_coordinate, z_coordinate])

    def _get_int_from_hillbert_coordinates(self, coordinates):
        coordinate_chunks = self._unpack_coordinates(coordinates)
        start, end = self._get_start_and_end_indices(len(coordinate_chunks), len(coordinates))
        chunks = [0] * len(coordinate_chunks)
        mask = 2 ** len(coordinates) - 1

        for chunk_index, current_chunk in enumerate(coordinate_chunks):
            gray_bit = self._get_gray_decoded(start, mask, current_chunk)
            chunks[chunk_index] = gray_bit
            start, end = self._get_child_start_and_end_indices(start, end, mask, gray_bit)

        return self._pack_index(chunks, len(coordinates))

    def _unpack_coordinates(self, coordinates):
        return [self._unpack_coordinate(coordinate) for coordinate in coordinates]

    def _unpack_coordinate(self, coordinate):
        return [int(bit) for bit in bin(coordinate)[2:].zfill(8)]

    def _get_start_and_end_indices(self, chunk_length, coordinate_length):
        return 0, chunk_length * coordinate_length

    def _get_gray_decoded(self, start, mask, current_chunk):
        gray_bit = current_chunk[start] ^ (current_chunk[start] >> 1)
        return gray_bit & mask

    def _get_child_start_and_end_indices(self, start, end, mask, gray_bit):
        return start + gray_bit, end - gray_bit

    def _pack_index(self, chunks, coordinate_length):
        index = 0
        for chunk in chunks:
            index = (index << coordinate_length) | chunk
        return index

def sort_colors(args):
    input_file = args.input_file
    color_format = args.format
    sorting_algorithm = args.sorting_algorithm
    output_file = args.output_file
    direction = args.direction

    with open(input_file, 'r') as file:
        colors_with_names = []
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if line:
                try:
                    color, name = parse_input_line(line)
                    colors_with_names.append((color, name))
                except Exception as e:
                    print(f"Error parsing line {line_number}: {line}")
                    print(f"Exception: {e}")

    if sorting_algorithm == 'HSV':
        colors_with_names.sort(key=lambda x: colorsys.rgb_to_hsv(*(c / 255.0 for c in x[0])))
    elif sorting_algorithm == 'HSL':
        colors_with_names.sort(key=lambda x: colorsys.rgb_to_hls(*(c / 255.0 for c in x[0])))
    elif sorting_algorithm == 'luminosity':
        colors_with_names.sort(key=lambda x: lum(*x[0]))
    elif sorting_algorithm == 'step':
        colors_with_names.sort(key=lambda x: step(*x[0], repetitions=8))
    elif sorting_algorithm == 'step-alternated':
        colors_with_names.sort(key=lambda x: step_alternated(*x[0], repetitions=8))
    elif sorting_algorithm == 'hilbert':
        hillbert_calculator = HillbertIndexCalculator()
        colors_with_names.sort(key=lambda x: hillbert_calculator.calculate(x[0]))
    else:
        colors_with_names.sort(key=lambda x: x[0])

    if direction == 'desc':
        colors_with_names.reverse()

    formatted_colors_with_names = [(format_color(color, color_format), name) for color, name in colors_with_names]
    
    for color, name in formatted_colors_with_names:
        print(f"{color} {name}")

    if output_file:
        with open(output_file, 'w') as f:
            for color, name in formatted_colors_with_names:
                f.write(f'{color} {name}\n')
        print(f"Sorted colors have been written to {output_file}")

def parse_input_line(line: str) -> Tuple[Tuple[int, int, int], str]:
    parts = line.split(maxsplit=1)
    color_str = parts[0]
    name = parts[1].strip() if len(parts) > 1 else ""

    if color_str.startswith('HSL') or color_str.startswith('RGB'):
        close_paren_index = line.find(')')
        if (close_paren_index != -1):
            color_str = line[:close_paren_index + 1]
            name = line[close_paren_index + 1:].strip()
        else:
            color_str = line
            name = ""

    try:
        color = parse_input_color(color_str)
    except ValueError as e:
        raise ValueError(f"Error parsing color '{color_str}': {e}")

    return color, name

def convert_colors(args):
    input_file = args.input_file
    output_format = args.format
    output_file = args.output_file

    with open(input_file, 'r') as file:
        colors_with_names = [parse_input_line(line) for line in file.readlines() if line.strip()]

    formatted_colors_with_names = [(format_color(color, output_format), name) for color, name in colors_with_names]
    for color, name in formatted_colors_with_names:
        print(f"{color} {name}")

    if output_file:
        with open(output_file, 'w') as f:
            for color, name in formatted_colors_with_names:
                f.write(f'{color} {name}\n')

def pack_colors_clr(args):
    mapping = {}
    text = open(args.input_file, 'r', encoding='utf-8').read()
    if text.lstrip().startswith('{'):
        obj = json.loads(text)
        mapping = {name: val for name, val in obj.items()}
    else:
        for line in text.splitlines():
            if not line.strip(): continue
            rgb, name = parse_input_line(line)
            mapping[name] = "{:02X}{:02X}{:02X}".format(*rgb)

    base = os.path.splitext(os.path.basename(args.output_file))[0]
    nsclr = NSColorList.alloc().initWithName_(base)

    for idx, (name, hex6) in enumerate(mapping.items()):
        h = hex6.lstrip('#')
        if len(h) < 6:
            h = ''.join(c*2 for c in h)
        r, g, b = [int(h[i:i+2], 16)/255.0 for i in (0,2,4)]
        ns = NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1.0)
        nsclr.insertColor_key_atIndex_(ns, name, idx)

    nsclr.writeToFile_(args.output_file)
    print(f".clr file written to {args.output_file}")

def unpack_colors_clr(args):
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    try:
        nsclr = NSColorList.alloc().initWithName_fromFile_(base, args.input_file)
    except Exception as e:
        print(f"Error: Unable to read CLR. {e}")
        return

    extracted = []
    for key in nsclr.allKeys():
        c = nsclr.colorWithKey_(key)
        r = int(c.redComponent()   * 255)
        g = int(c.greenComponent() * 255)
        b = int(c.blueComponent()  * 255)
        extracted.append(((r, g, b), key))

    if not extracted:
        print("No colors were extracted from the file.")
        return

    formatted = [(format_color(rgb, args.format), name) for rgb, name in extracted]
    for col, name in formatted:
        print(f"{col} {name}")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for col, name in formatted:
                f.write(f"{col} {name}\n")
        print(f"Colors written to {args.output_file}")

def pack_colors_aco(args):
    """Packs colors into a `.aco` file (version 2 only).

    Args:
        args: Command-line arguments containing input and output file paths.
    """
    input_file = args.input_file
    output_file = args.output_file

    def write_word(fp, value):
        fp.write(value.to_bytes(2, 'big'))

    def write_dword(fp, value):
        fp.write(value.to_bytes(4, 'big'))

    def write_string(fp, string):
        encoded = string.encode('utf-16-be')
        fp.write(encoded)
        fp.write(b'\x00\x00')

    colors = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                color, name = parse_input_line(line.strip())
                colors.append((color, name))

    with open(output_file, 'wb') as fp:
        write_word(fp, 1)
        write_word(fp, len(colors))
        for color, _ in colors:
            write_word(fp, 0)
            for component in color:
                write_word(fp, component * 257)
            write_word(fp, 0)

        write_word(fp, 2)
        write_word(fp, len(colors))
        for color, name in colors:
            write_word(fp, 0)
            for component in color:
                write_word(fp, component * 257)
            write_word(fp, 0)
            write_dword(fp, len(name) + 1)
            write_string(fp, name)

    print(f"ACO file has been written to {output_file}")

def unpack_colors_aco(args):
    input_file = args.input_file
    output_format = args.format
    output_file = args.output_file

    def read_word(fp):
        bytes = fp.read(2)
        if len(bytes) < 2:
            return -1
        return (bytes[0] << 8) | bytes[1]

    def must_read_word(fp):
        word = read_word(fp)
        if word == -1:
            raise ValueError("Unexpected end of file!")
        return word

    def read_int32(fp):
        bytes = fp.read(4)
        if len(bytes) < 4:
            raise ValueError("Unexpected end of file!")
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]

    def read_string(fp, length):
        buffer = fp.read(length * 2)
        return buffer.decode('utf-16be').rstrip('\x00')

    def component_to_hex(c):
        hex_value = hex(c)[2:]
        return hex_value.zfill(2)

    def rgb_to_hex(r, g, b):
        return f"{r:02X}{g:02X}{b:02X}"

    def convert_color(fp, ver):
        cspace = must_read_word(fp)
        value1 = must_read_word(fp)
        value2 = must_read_word(fp)
        value3 = must_read_word(fp)
        value4 = must_read_word(fp)

        if cspace == 0:
            r = value1 // 256
            g = value2 // 256
            b = value3 // 256
            color = (r, g, b)
        elif cspace == 8:
            gray = value1 // 39.0625
            color = (int(gray), int(gray), int(gray))
        else:
            return None

        name = None
        if ver == 2:
            name_length = read_int32(fp)
            name = read_string(fp, name_length)

        return color, name

    def read_aco(fp):
        ver = must_read_word(fp)
        if ver not in [1, 2]:
            raise ValueError(f"Unknown ACO file version {ver}")

        num_colors = must_read_word(fp)
        colors = []
        for _ in range(num_colors):
            color = convert_color(fp, ver)
            if color:
                colors.append(color)

        if ver == 1:
            try:
                ver = must_read_word(fp)
                if ver == 2:
                    num_colors = must_read_word(fp)
                    colors = []
                    for _ in range(num_colors):
                        color = convert_color(fp, ver)
                        if color:
                            colors.append(color)
            except ValueError:
                pass

        return colors

    try:
        with open(input_file, 'rb') as fp:
            colors = read_aco(fp)

        unique_colors = {}
        for color, name in colors:
            formatted_color = format_color(color, output_format)
            if formatted_color not in unique_colors:
                unique_colors[formatted_color] = name

        formatted_colors_with_names = [(formatted_color, name) for formatted_color, name in unique_colors.items()]
        for color, name in formatted_colors_with_names:
            print(f"{color} {name}")

        if output_file:
            with open(output_file, 'w') as f:
                for color, name in formatted_colors_with_names:
                    f.write(f'{color} {name}\n')
            print(f"Colors have been written to {output_file}")

    except Exception as e:
        print(f"Error: Unable to read the ACO file. {str(e)}")

def pack_colors_ase(args):
    def make_ase(palettes):
        output = io.BytesIO()

        total_colors = sum(len(palette["colors"]) for palette in palettes)
        num_palettes = len(palettes)

        output.write(b"ASEF")
        output.write(struct.pack(">HH", 1, 0))
        output.write(struct.pack(">I", total_colors + (num_palettes * 2)))

        for palette in palettes:
            output.write(struct.pack(">H", 0xC001))

            title = palette["title"].encode('utf-16be') + b'\x00\x00'
            buffer = struct.pack(">H", len(title) // 2)
            buffer += title

            output.write(struct.pack(">I", len(buffer)))
            output.write(buffer)

            for color in palette["colors"]:
                output.write(struct.pack(">H", 1))

                title = color[1].encode('utf-16be') + b'\x00\x00'
                buffer = struct.pack(">H", len(title) // 2)
                buffer = struct.pack(">H", len(title) // 2)
                buffer += title

                r, g, b = [int(color[0][i:i+2], 16) / 255 for i in (0, 2, 4)]
                buffer += b"RGB "
                buffer += struct.pack(">fff", r, g, b)
                buffer += struct.pack(">H", 0)

                output.write(struct.pack(">I", len(buffer)))
                output.write(buffer)

            output.write(struct.pack(">H", 0xC002))
            output.write(struct.pack(">I", 0))

        return output.getvalue()

    input_file = args.input_file
    output_file = args.output_file

    try:
        with open(input_file, 'r') as file:
            colors = [parse_input_line(line.strip()) for line in file if line.strip()]

        palette = {
            "title": "Color Palette",
            "colors": [(convert_to_hex(color)[1:], name or f"Color {i+1}") for i, (color, name) in enumerate(colors)]
        }

        ase_data = make_ase([palette])

        with open(output_file, 'wb') as ase_file:
            ase_file.write(ase_data)

        print(f"Colors have been packed into ASE file: {output_file}")

    except Exception as e:
        print(f"Error: Unable to pack colors into ASE file. {str(e)}")
        print("Make sure the input file contains valid color data.")

def unpack_colors_ase(args):
    input_file = args.input_file
    output_format = args.format
    output_file = args.output_file

    def read_ase_string(data, offset):
        length = struct.unpack(">H", data[offset:offset + 2])[0] * 2
        offset += 2
        string = data[offset:offset + length].decode('utf-16be').rstrip('\x00')
        offset += length
        return string, offset

    try:
        with open(input_file, 'rb') as ase_file:
            ase_data = ase_file.read()

        offset = 0
        signature = ase_data[offset:offset + 4]
        offset += 4
        if signature != b"ASEF":
            raise ValueError("Not a valid ASE file")

        version = struct.unpack(">HH", ase_data[offset:offset + 4])
        offset += 4
        total_blocks = struct.unpack(">I", ase_data[offset:offset + 4])[0]
        offset += 4

        colors = []
        while offset < len(ase_data):
            block_type = struct.unpack(">H", ase_data[offset:offset + 2])[0]
            offset += 2
            block_length = struct.unpack(">I", ase_data[offset:offset + 4])[0]
            offset += 4

            if block_type == 0xC001:
                group_name, offset = read_ase_string(ase_data, offset)
            elif block_type == 0xC002:
                continue
            elif block_type == 1:
                color_name, offset = read_ase_string(ase_data, offset)
                color_model = ase_data[offset:offset + 4].decode('ascii')
                offset += 4

                if color_model == "RGB ":
                    r, g, b = struct.unpack(">fff", ase_data[offset:offset + 12])
                    offset += 12
                    color = (int(r * 255), int(g * 255), int(b * 255))
                else:
                    offset += block_length - 4 - len(color_name) * 2 - 2

                colors.append((color, color_name))
                offset += 2

        formatted_colors_with_names = [(format_color(color, output_format), name) for color, name in colors]
        for color, name in formatted_colors_with_names:
            print(f"{color} {name}")

        if output_file:
            with open(output_file, 'w') as f:
                for color, name in formatted_colors_with_names:
                    f.write(f'{color} {name}\n')
            print(f"Colors have been written to {output_file}")

    except Exception as e:
        print(f"Error: Unable to read the ASE file. {str(e)}")

def unpack_colors(args):
    file_extension = args.input_file.split('.')[-1].lower()
    unpack_functions = {
        'clr': unpack_colors_clr,
        'ase': unpack_colors_ase,
        'aco': unpack_colors_aco
    }
    
    if file_extension in unpack_functions:
        unpack_functions[file_extension](args)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
def pack_colors(args):
    file_format = args.format.lower()
    pack_functions = {
        'clr': pack_colors_clr,
        'ase': pack_colors_ase,
        'aco': pack_colors_aco
    }
    
    if file_format in pack_functions:
        pack_functions[file_format](args)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def main():
    parser = argparse.ArgumentParser(description='Command-line tool for sorting, converting, packing, and unpacking colors with multiple formats.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_sort = subparsers.add_parser('sort', help='Sort colors based on RGB values.')
    parser_sort.add_argument('--sorting-algorithm', default='RGB', choices=['RGB', 'HSV', 'HSL', 'luminosity', 'step', 'step-alternated', 'hilbert'], help='Sorting algorithm to use for colors. Options: RGB, HSV, HSL, luminosity, step, step-alternated, hilbert.')
    parser_sort.add_argument('--input-file', type=str, required=True, help='File containing colors, one per line, in either RGB, HEX or HSL format.')
    parser_sort.add_argument('--format', type=str, required=True, choices=['RGB', 'HEX', 'HSL'], help='Format of the output sorted colors. Options: RGB, HEX, HSL.')
    parser_sort.add_argument('--output-file', type=str, help='File to save the sorted and formatted colors.')
    parser_sort.add_argument('--direction', default='asc', choices=['asc', 'desc'], help='Sort direction of the colors. Options: asc (ascending), desc (descending).')
    parser_sort.set_defaults(func=sort_colors)

    parser_convert = subparsers.add_parser('convert', help='Convert colors from one format to another.')
    parser_convert.add_argument('--input-file', type=str, required=True, help='File containing colors, one per line, in either RGB, HEX or HSL format.')
    parser_convert.add_argument('--format', type=str, required=True, choices=['RGB', 'HEX', 'HSL'], help='Format of the output converted colors. Options: RGB, HEX, HSL.')
    parser_convert.add_argument('--output-file', type=str, help='File to save the converted colors.')
    parser_convert.set_defaults(func=convert_colors)

    parser_pack = subparsers.add_parser('pack', help='Pack colors from a text file into a color file.')
    parser_pack.add_argument('--input-file', type=str, required=True, help='File containing colors, one per line, in either RGB, HEX or HSL format.')
    parser_pack.add_argument('--format', type=str, required=True, choices=['clr', 'ase', 'aco'], help='Format of the output packed colors. Options: CLR, ASE, ACO.')
    parser_pack.add_argument('--output-file', type=str, required=True, help='File to save the packed colors.')
    parser_pack.set_defaults(func=pack_colors)

    parser_unpack = subparsers.add_parser('unpack', help='Unpack colors from a color file into a text file.')
    parser_unpack.add_argument('--input-file', type=str, required=True, help='Color file containing packed colors.')
    parser_unpack.add_argument('--format', type=str, required=True, choices=['RGB', 'HEX', 'HSL'], help='Format of the output unpacked colors. Options: RGB, HEX, HSL.')
    parser_unpack.add_argument('--output-file', type=str, help='File to save the unpacked colors.')
    parser_unpack.set_defaults(func=unpack_colors)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == '__main__':
    main()