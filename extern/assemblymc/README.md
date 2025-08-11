# AssemblyMC Binary

This directory vendors the `AssemblyMC` Monte Carlo solver from the Cronin Group's [Paper-AssemblyTreeOfLife](https://github.com/croningp/Paper-AssemblyTreeOfLife) project.

## Obtaining the binary

The upstream repository distributes a prebuilt Windows executable as `AssemblyMC.zip`. Download that file and rename it to `AssemblyMC.exe` **without unzipping**. Place the renamed executable **and the accompanying `libinchi.dll`** in the `bin/` directory beside this README. These files are not tracked in version control; you must obtain them from the upstream project.

## Running

On Windows, invoke the executable directly:

```powershell
AssemblyMC.exe --help
```

On Unix-like systems, install [Wine](https://www.winehq.org/) and use the wrapper script:

```bash
./AssemblyMC --help
```

The wrapper forwards all arguments to `AssemblyMC.exe` via `wine` and prints helpful instructions if Wine or the executable are missing.

## Building from source

1. Clone the upstream repository:
   ```bash
   git clone https://github.com/croningp/Paper-AssemblyTreeOfLife
   ```
2. The C++ sources live under `AssemblyMC_cpp_sc/`.
3. Follow the upstream instructions to build with Visual Studio 2019 or later. The build expects the bundled InChI library found under `AssemblyMC_cpp_sc/INCHI_API`.

## Permission and licensing

The `AssemblyMC` binary is the intellectual property of the Cronin Group. Please consult the upstream project and obtain appropriate permission before using the executable in other contexts.
