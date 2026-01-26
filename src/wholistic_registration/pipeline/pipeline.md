## WHOLISTIC Registration Pipeline

- Define data path
- Extract metadata from data (pixel resolution, frame rate, data type)
- Convert data to necessary format (if needed)
- Define registration parameters (this should include the version of the code used)
- Save registration parameters to metafile (along with metadata of data)
- Load these parameters from save file 
- Run registration (ideally there should be an estimate of CPU/GPU resources needed based on data volume)
- If there are multiple references, save references into one tif file to inspect visually for drift
- Apply registration to both channels
- Save registered data with metadata from the original file (pixel resolution, frame rate, data type)
- Create downsampled version of registered data (calcium and membrane) to inspect visually results
- Create 4D mask to definie well/poorly registered areas (ie.: what pixels at what time can be trusted)
- 
... 

## Results folder

- registered data
    - calcium
    - membrane
    - references
        - labeled with timepoints that are associated with the reference
    - mask
        - folder of tiffs (one per timepoint)
        - channels: mask

- downsampled data
    - calcium
        - folder of tiffs (one per downsampled timepoint)
        - contains downsampled calcium data
        - channels: original data, registered data, reference used for that timepoint
    - membrane
        - folder of tiffs (one per downsampled timepoint)
        - contains downsampled membrane data
        - channels: original data, registered data, reference used for that timepoint
    - downsampled mask
        - folder of tiffs (one per downsampled timepoint)
        - channels: registered calcium data, registered membrane data, mask (all downsampled data)