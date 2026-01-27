// Set starting directory and prompt user
startPath = "/nrs/ahrens/Virginia_nrs/wVT/";
call("ij.io.OpenDialog.setDefaultDirectory", startPath);
dir = getDirectory("Select folder containing TIFF volumes");

// Get list of files
list = getFileList(dir);
nTotalFiles = list.length;

if (nTotalFiles == 0) {
    exit("No files found in selected folder");
}

// Open first file to get dimensions
open(dir + list[0]);
getDimensions(width, height, nChannels, zDepth, frames);
slicesPerFile = nSlices();
print("First file: " + width + "x" + height + ", nSlices=" + slicesPerFile + ", nChannels=" + nChannels + ", zDepth=" + zDepth + ", frames=" + frames);
close();

// Get loading options from user
Dialog.create("Loading Options");
Dialog.addMessage("Found " + nTotalFiles + " files");
Dialog.addMessage("First file: " + width + "x" + height + ", " + slicesPerFile + " slices (" + nChannels + "C x " + zDepth + "Z)");
Dialog.addCheckbox("Load all files", false);
Dialog.addNumber("Or load every N files:", 10);
Dialog.show();
loadAll = Dialog.getCheckbox();
stepSize = Dialog.getNumber();

if (loadAll) {
    stepSize = 1;
}

// Count how many we'll load
nTimepoints = floor(nTotalFiles / stepSize);
if (nTotalFiles % stepSize != 0) nTimepoints = nTimepoints + 1;

print("Loading " + nTimepoints + " files (every " + stepSize + " of " + nTotalFiles + ")");
print("Each file: " + nChannels + "C x " + zDepth + "Z = " + slicesPerFile + " slices");

// Open files
setBatchMode(true);
count = 0;
for (i = 0; i < nTotalFiles; i += stepSize) {
    showProgress(count, nTimepoints);
    open(dir + list[i]);
    if (count == 0) {
        rename("Stack");
    } else {
        run("Concatenate...", "  title=Stack image1=Stack image2=[" + getTitle() + "] image3=[-- None --]");
    }
    count++;
}
setBatchMode(false);

// Reshape to hyperstack
if (zDepth > 1 || nChannels > 1) {
    run("Stack to Hyperstack...", "order=xyczt(default) channels=" + nChannels + " slices=" + zDepth + " frames=" + count + " display=Composite");
}

// Auto contrast for each channel
for (c = 1; c <= nChannels; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

print("Done! Loaded " + count + " files.");