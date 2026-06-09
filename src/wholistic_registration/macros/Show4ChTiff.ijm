// ============================================================
// Show4ChTiff.ijm
//
// Open the 4-channel wholistic registration output as a composite
// hyperstack in Fiji.  The user selects a root folder that contains
// four sub-folders:
//
//   raw_moving_mem/
//   projected_mem/
//   raw_moving_sparseCell/
//   projected_sparseCell/
//
// Z depth and T count are read automatically from the TIFF files.
//
// Channel order:
//   C1: raw_moving_mem           (grey)
//   C2: projected_mem            (green)
//   C3: raw_moving_sparseCell    (magenta)
//   C4: projected_sparseCell     (cyan)
//
// Usage:
//   Drag this file onto the Fiji toolbar, or
//   Plugins > Macros > Run... and select this file.
//   Then pick the root folder in the dialog.
// ============================================================

setBatchMode(true);

// --------------------------------------------------
// 1. User selects the root folder
// --------------------------------------------------
base_dir = getDirectory("Select the root folder containing the 4 channel sub-folders");
if (base_dir == "")
    exit("No folder selected.");

// --------------------------------------------------
// 2. Channel definitions (fixed sub-folder names)
// --------------------------------------------------
ch_folders  = newArray(
    "raw_moving_mem",
    "projected_mem",
    "raw_moving_sparseCell",
    "projected_sparseCell"
);
ch_labels = newArray(
    "C1-raw_mem",
    "C2-proj_mem",
    "C3-raw_sparse",
    "C4-proj_sparse"
);
n_ch = 4;

// --------------------------------------------------
// 3. Auto-detect Z depth & T count from the TIFFs
// --------------------------------------------------
first_ch_dir = base_dir + ch_folders[0] + "/";

// List and sort TIFF files in the first channel folder
raw_list = getFileList(first_ch_dir);
tif_list = newArray();
for (i = 0; i < raw_list.length; i++) {
    if (endsWith(raw_list[i], ".tif"))
        tif_list = Array.concat(tif_list, raw_list[i]);
}
Array.sort(tif_list);

if (tif_list.length == 0)
    exit("No TIFF files found in " + first_ch_dir);

n_frames = tif_list.length;

// Open the first TIFF to read its Z depth
open(first_ch_dir + tif_list[0]);
getDimensions(w, h, c, z_depth, t_depth);
close();

n_zslices = z_depth;

print("Auto-detected:  " + n_frames + " T frames  ×  " + n_zslices + " Z slices");
print("Root folder:     " + base_dir);

// --------------------------------------------------
// 4. Auto-detect file prefix for each channel
//    (strip the trailing 6-digit number + .tif)
// --------------------------------------------------
ch_prefixes = newArray(n_ch);
for (c = 0; c < n_ch; c++) {
    ch_dir = base_dir + ch_folders[c] + "/";
    ch_list = getFileList(ch_dir);
    ch_tifs = newArray();
    for (i = 0; i < ch_list.length; i++) {
        if (endsWith(ch_list[i], ".tif"))
            ch_tifs = Array.concat(ch_tifs, ch_list[i]);
    }
    Array.sort(ch_tifs);

    if (ch_tifs.length == 0)
        exit("No TIFF files in " + ch_dir);

    // filename example:  vol_F260517_raw_mem_000000.tif
    // Strip last 10 chars  (6 digits + ".tif")
    fname = ch_tifs[0];
    prefix = substring(fname, 0, lengthOf(fname) - 10);
    ch_prefixes[c] = prefix;

    print("  Ch" + (c+1) + " prefix: " + prefix + "  (" + ch_tifs.length + " files)");
}

// --------------------------------------------------
// 5. Open each channel as a virtual image sequence
// --------------------------------------------------
for (c = 0; c < n_ch; c++) {
    folder    = base_dir + ch_folders[c] + "/";
    prefix    = ch_prefixes[c];
    label     = ch_labels[c];
    first_tif = folder + ch_prefixes[c] + "000000.tif";

    if (!File.exists(first_tif)) {
        // Try without zero-padded frame number as fallback
        print("[WARN] " + first_tif + " not found, scanning...");
        fallback_list = getFileList(folder);
        found = false;
        for (k = 0; k < fallback_list.length; k++) {
            if (endsWith(fallback_list[k], ".tif")) {
                first_tif = folder + fallback_list[k];
                found = true;
                break;
            }
        }
        if (!found)
            exit("Cannot find any TIFF in " + folder);
    }

    print("[" + (c+1) + "/" + n_ch + "] Opening: " + label);

    run("Image Sequence...",
        "open=" + first_tif +
        " sort use_virtual");

    // Set hyperstack dimensions:
    //   channels=1  slices = Z depth   frames = T count
    run("Properties...",
        "channels=1 " +
        "slices=" + n_zslices + " " +
        "frames=" + n_frames + " " +
        "unit=pixel");

    rename(label);
}

// --------------------------------------------------
// 6. Merge 4 channels into one composite hyperstack
// --------------------------------------------------
print("Merging 4 channels...");

run("Merge Channels...",
    "c1=" + ch_labels[0] + " " +
    "c2=" + ch_labels[1] + " " +
    "c3=" + ch_labels[2] + " " +
    "c4=" + ch_labels[3] + " " +
    "create");

rename("F260517_4ch");

// --------------------------------------------------
// 7. Display settings
// --------------------------------------------------
selectWindow("F260517_4ch");

Stack.setChannel(1);   // C1: grey  (default)
Stack.setChannel(2);   run("Green");
Stack.setChannel(3);   run("Magenta");
Stack.setChannel(4);   run("Cyan");

Stack.setChannel(1);
Stack.setDisplayMode("composite");

// Auto contrast per channel
for (c = 1; c <= n_ch; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

Stack.setPosition(1, n_zslices / 2, n_frames / 2);

setBatchMode(false);

print("========================================");
print("Done!");
print("  Image:  F260517_4ch");
print("  Dims:   " + n_zslices + "Z  ×  " + n_frames + "T  ×  4C");
print("----------------------------------------");
print("  Alt + Mousewheel  →  change Z");
print("  Ctrl + Mousewheel →  change T");
print("========================================");
