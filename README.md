### Current Color Palettes

| Name | macOS CLR | Adobe ASE | Adobe ACO |
| :--- | :---: | :---: | :---: |
| AWS 2021 | [CLR](CLR/AWS%202021.clr?raw=1) | [ASE](ASE/AWS%202021.ase?raw=1) | [ACO](ACO/AWS%202021.aco?raw=1) |
| Apple | [CLR](CLR/Apple.clr?raw=1) | [ASE](ASE/Apple.ase?raw=1) | [ACO](ACO/Apple.aco?raw=1) |
| Colors of Japan | [CLR](CLR/Colors%20of%20Japan.clr?raw=1) | [ASE](ASE/Colors%20of%20Japan.ase?raw=1) | [ACO](ACO/Colors%20of%20Japan.aco?raw=1) |
| Open Color | [CLR](CLR/Open%20Color.clr?raw=1) | [ASE](ASE/Open%20Color.ase?raw=1) | [ACO](ACO/Open%20Color.aco?raw=1) |
| Red Colours | [CLR](CLR/Red%20Colours.clr?raw=1) | [ASE](ASE/Red%20Colours.ase?raw=1) | [ACO](ACO/Red%20Colours.aco?raw=1) |
| Orange Colours | [CLR](CLR/Orange%20Colours.clr?raw=1) | [ASE](ASE/Orange%20Colours.ase?raw=1) | [ACO](ACO/Orange%20Colours.aco?raw=1) |
| Yellow Colours | [CLR](CLR/Yellow%20Colours.clr?raw=1) | [ASE](ASE/Yellow%20Colours.ase?raw=1) | [ACO](ACO/Yellow%20Colours.aco?raw=1) |
| Green Colours | [CLR](CLR/Green%20Colours.clr?raw=1) | [ASE](ASE/Green%20Colours.ase?raw=1) | [ACO](ACO/Green%20Colours.aco?raw=1) |
| Blue Colours | [CLR](CLR/Blue%20Colours.clr?raw=1) | [ASE](ASE/Blue%20Colours.ase?raw=1) | [ACO](ACO/Blue%20Colours.aco?raw=1) |
| Purple Colours | [CLR](CLR/Purple%20Colours.clr?raw=1) | [ASE](ASE/Purple%20Colours.ase?raw=1) | [ACO](ACO/Purple%20Colours.aco?raw=1) |
| Pink Colours | [CLR](CLR/Pink%20Colours.clr?raw=1) | [ASE](ASE/Pink%20Colours.ase?raw=1) | [ACO](ACO/Pink%20Colours.aco?raw=1) |
| Brown Colours | [CLR](CLR/Brown%20Colours.clr?raw=1) | [ASE](ASE/Brown%20Colours.ase?raw=1) | [ACO](ACO/Brown%20Colours.aco?raw=1) |
| Black Colours | [CLR](CLR/Black%20Colours.clr?raw=1) | [ASE](ASE/Black%20Colours.ase?raw=1) | [ACO](ACO/Black%20Colours.aco?raw=1) |
| Gray Colours | [CLR](CLR/Gray%20Colours.clr?raw=1) | [ASE](ASE/Gray%20Colours.ase?raw=1) | [ACO](ACO/Gray%20Colours.aco?raw=1) |
| White Colours | [CLR](CLR/White%20Colours.clr?raw=1) | [ASE](ASE/White%20Colours.ase?raw=1) | [ACO](ACO/White%20Colours.aco?raw=1) |

### About

This tool will do, via a handler, the following:
- it will sort colors in a file, based on one of the supported algorithms;
- it will convert the input file between HEX, RGB and HSL formats;
- it will pack a HEX, RGB or HSL format into ACO and ASE formats;
- it will unpack an ASE or ACO file into any of the HEX, RGB or HSL formats.

Install requirements:

```
python3 -m pip install -r requirements.txt
```

Due to `pyobjc` dependency, this script will work only MacOS.

**Usage:**

```
color_tool.py --help
usage: color_tool.py [-h] {sort,convert,pack,unpack} ...

Command-line tool for sorting, converting, packing, and unpacking colors with multiple formats.

positional arguments:
  {sort,convert,pack,unpack}
    sort                Sort colors based on RGB values.
    convert             Convert colors from one format to another.
    pack                Pack colors from a text file into a color file.
    unpack              Unpack colors from a color file into a text file.

optional arguments:
  -h, --help            show this help message and exit
```

**sort** handler

```
color_tool.py sort --help
usage: color_tool.py sort [-h] [--sorting-algorithm {RGB,HSV,HSL,luminosity,step,step-alternated,hilbert}] --input-file INPUT_FILE
                          --format {RGB,HEX,HSL} [--output-file OUTPUT_FILE] [--direction {asc,desc}]

optional arguments:
  -h, --help            show this help message and exit
  --sorting-algorithm {RGB,HSV,HSL,luminosity,step,step-alternated,hilbert}
                        Sorting algorithm to use for colors. Options: RGB, HSV, HSL, luminosity, step, step-alternated, hilbert.
  --input-file INPUT_FILE
                        File containing colors, one per line, in either RGB, HEX or HSL format.
  --format {RGB,HEX,HSL}
                        Format of the output sorted colors. Options: RGB, HEX, HSL.
  --output-file OUTPUT_FILE
                        File to save the sorted and formatted colors.
  --direction {asc,desc}
                        Sort direction of the colors. Options: asc (ascending), desc (descending).
```

**convert** handler:

```
color_tool.py convert --help
usage: color_tool.py convert [-h] --input-file INPUT_FILE --format {RGB,HEX,HSL} [--output-file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        File containing colors, one per line, in either RGB, HEX or HSL format.
  --format {RGB,HEX,HSL}
                        Format of the output converted colors. Options: RGB, HEX, HSL.
  --output-file OUTPUT_FILE
                        File to save the converted colors.
```

**pack** handler

```
color_tool.py pack --help
usage: color_tool.py pack [-h] --input-file INPUT_FILE --format {clr,ase,aco} --output-file OUTPUT_FILE

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        File containing colors, one per line, in either RGB, HEX or HSL format.
  --format {clr,ase,aco}
                        Format of the output packed colors. Options: CLR, ASE, ACO.
  --output-file OUTPUT_FILE
                        File to save the packed colors.
```

**unpack** handler

```
color_tool.py unpack --help
usage: color_tool.py unpack [-h] --input-file INPUT_FILE --format {RGB,HEX,HSL} [--output-file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        Color file containing packed colors.
  --format {RGB,HEX,HSL}
                        Format of the output unpacked colors. Options: RGB, HEX, HSL.
  --output-file OUTPUT_FILE
                        File to save the unpacked colors.
```

**Input File Sample**

```
<HEX_COLOR|RGB_COLOR|HSL_COLOR> <COLOR_NAME
```

```
#010203 HEX Color
RGB(56, 99, 79) Resene
HSL(39, 45%, 42%) Resene
```