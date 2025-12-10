// Prompt user for folder
dir = getDirectory("Select folder containing TIFF volumes");

// Get list of TIFF files
list = getFileList(dir);
tiffFiles = newArray();
for (i = 0; i < list.length; i++) {
    if (endsWith(toLowerCase(list[i]), ".tif") || endsWith(toLowerCase(list[i]), ".tiff")) {
        tiffFiles = Array.concat(tiffFiles, list[i]);
    }
}

nTimepoints = tiffFiles.length;
if (nTimepoints == 0) {
    exit("No TIFF files found in selected folder");
}

// Open first image to get dimensions
open(dir + tiffFiles[0]);
getDimensions(width, height, nChannels, zDepth, frames);
close();

print("Found " + nTimepoints + " files");
print("Each file: " + zDepth + " Z slices, " + nChannels + " channels");
print("Will create hyperstack: C=" + nChannels + ", Z=" + zDepth + ", T=" + nTimepoints);

// Import the full sequence
run("Image Sequence...", "open=[" + dir + "] sort");

// Reshape to hyperstack
run("Stack to Hyperstack...", "order=xyczt(default) channels=" + nChannels + " slices=" + zDepth + " frames=" + nTimepoints + " display=Composite");

// Auto contrast for each channel
for (c = 1; c <= nChannels; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

print("Done!");