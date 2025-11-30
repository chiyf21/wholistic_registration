% addpath(genpath("/home/cyf/wbi/Virginia/wei_code/wbi_code"));

%% file path
filePath = "/home/cyf/wbi/Virginia/f338/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2";
resPathName = "/home/cyf/wbi/Virginia/result_0906_15";
artiDataPath = "/home/cyf/wbi/Virginia/simulation_0906";

%% load data
motionPath = resPathName + "/motion";
fileLst = dir(motionPath + "/dat*");
reader = bfGetReader(convertStringsToChars(filePath));
[X, Y, Z_total, ~, ~, option.zRatio] = readMeta(reader); % origin z dimension
dat_org = readOneFrame_single(reader, 1, 2);

%% check z
if Z_total < 13
    error('orgin z dimension(%d) less than 13，couldnt been cropped to 13 slices', Z_total);
elseif Z_total > 13
    Z = 13; 
else
    Z = Z_total;
end

%% define crop size
crop_size = [256, 256, Z]; % [X, Y, Z]
if X < crop_size(1) || Y < crop_size(2)
    error('origin data shape (%d×%d) smaller than crop shape(%d×%d)', X, Y, crop_size(1), crop_size(2));
end

%% DEFAULT
num_repeats = 20;  % rep num
d_NoiseLevel = 5;
d_R = 20;
d_Amp = 5;

lst_Noise = 1.4.^(0:9);
lst_R = 5:20;
lst_Amp = 1:10;

%% create main content
for rep = 1:num_repeats
    rep_dir = sprintf("%s/rep%02d", artiDataPath, rep);
    mkdir(rep_dir);
    
   
    mkdir(sprintf("%s/R", rep_dir));
    mkdir(sprintf("%s/Amp", rep_dir));
    mkdir(sprintf("%s/Noise", rep_dir));
end

%% reliable map
[pMap, zMap] = ReliabilityZScore(dat_org, 5);
save(artiDataPath + "/opt.mat", "dat_org", "zMap", ...
    "d_NoiseLevel", "d_Amp", "d_R", ...
    "lst_Noise", "lst_Amp", "lst_R", "-v7.3");
dist = 15;   %
%% generate repeat data
%% generate repeat data
for rep = 1:num_repeats
    tic
    fprintf('===== beginning to generate %d/%d  =====\n', rep, num_repeats);
    
    rep_dir = sprintf("%s/rep%02d", artiDataPath, rep);
    
    %% region to crop
    x_start = 198;
    y_start = 1580;
    z_start = 1; 
    crop_range_x = x_start:(x_start + crop_size(1) - 1);
    crop_range_y = y_start:(y_start + crop_size(2) - 1);
    crop_range_z = z_start:(z_start + crop_size(3) - 1);
    
    fprintf(' rep%d crop region: X=%d-%d, Y=%d-%d, Z=%d-%d\n', ...
            rep, x_start, x_start+crop_size(1)-1, ...
            y_start, y_start+crop_size(2)-1, ...
            z_start, z_start+crop_size(3)-1);
    

    save(sprintf("%s/crop_info.mat", rep_dir), ...
         "crop_range_x", "crop_range_y", "crop_range_z");
    
    %% generate R data
    Amp_art = d_Amp;
    NoiseLevel = d_NoiseLevel;
    for cnt = 1:length(lst_R)
        art_R = lst_R(cnt);
        [motion_X, motion_Y, motion_Z] = generateMotion_Simple2D_v3(resPathName, reader, art_R, Amp_art, option.zRatio);
        
        % generate rigid translation
        direction = randn(2, 1);
        shift = round(direction / norm(direction) * dist);
        shift = shift(:)';
        motion_X = motion_X + shift(1);
        motion_Y = motion_Y + shift(2);
        
        motion_current_real = cat(4, motion_X, motion_Y, motion_Z);
        dat_mov_raw = correctMotion_Wei_v2(dat_org, -motion_current_real);
        dat_ref_raw = correctMotion_Wei_v2(dat_mov_raw, motion_current_real);
        dat_mov = dat_mov_raw + randn(X, Y, Z_total) * NoiseLevel;
        dat_ref = dat_ref_raw + randn(X, Y, Z_total) * NoiseLevel;
        
        
        % Crop data to shared region
        dat_mov = dat_mov(crop_range_x, crop_range_y, crop_range_z);
        dat_ref = dat_ref(crop_range_x, crop_range_y, crop_range_z);
        motion_current_real = motion_current_real(crop_range_x, crop_range_y, crop_range_z, :);
        
        save(sprintf("%s/R/%d.mat", rep_dir, cnt), ...
             "dat_mov", "dat_ref", "motion_current_real");
    end
    
    %% Generate Amp parameter data (same region, different motion fields)
    art_R = d_R;
    NoiseLevel = d_NoiseLevel;
    for cnt = 1:length(lst_Amp)
        Amp_art = lst_Amp(cnt);
        [motion_X, motion_Y, motion_Z] = generateMotion_Simple2D_v3(resPathName, reader, art_R, Amp_art, option.zRatio);
      
        % Generate random rigid translation
        direction = randn(2, 1);
        shift = round(direction / norm(direction) * dist);
        shift = shift(:)';
        motion_X = motion_X + shift(1);
        motion_Y = motion_Y + shift(2);

        motion_current_real = cat(4, motion_X, motion_Y, motion_Z);
        dat_mov_raw = correctMotion_Wei_v2(dat_org, -motion_current_real);
        dat_ref_raw = correctMotion_Wei_v2(dat_mov_raw, motion_current_real);
        dat_mov = dat_mov_raw + randn(X, Y, Z_total) * NoiseLevel;
        dat_ref = dat_ref_raw + randn(X, Y, Z_total) * NoiseLevel;
        
        
        % Crop data to shared region
        dat_mov = dat_mov(crop_range_x, crop_range_y, crop_range_z);
        dat_ref = dat_ref(crop_range_x, crop_range_y, crop_range_z);
        motion_current_real = motion_current_real(crop_range_x, crop_range_y, crop_range_z, :);

        
        save(sprintf("%s/Amp/%d.mat", rep_dir, cnt), ...
             "dat_mov", "dat_ref", "motion_current_real");
    end
    
    %% Generate Noise parameter data (same region, different motion fields)
    art_R = d_R;
    Amp_art = d_Amp;
    for cnt = 1:length(lst_Noise)
        NoiseLevel = lst_Noise(cnt);
        [motion_X, motion_Y, motion_Z] = generateMotion_Simple2D_v3(resPathName, reader, art_R, Amp_art, option.zRatio);
        
        % Generate random rigid translation
        direction = randn(2, 1);
        shift = round(direction / norm(direction) * dist);
        shift = shift(:)';
        motion_X = motion_X + shift(1);
        motion_Y = motion_Y + shift(2);

        motion_current_real = cat(4, motion_X, motion_Y, motion_Z);
        dat_mov_raw = correctMotion_Wei_v2(dat_org, -motion_current_real);
        dat_ref_raw = correctMotion_Wei_v2(dat_mov_raw, motion_current_real);
        dat_mov = dat_mov_raw + randn(X, Y, Z_total) * NoiseLevel;
        dat_ref = dat_ref_raw + randn(X, Y, Z_total) * NoiseLevel;
        
        
        % Crop data to shared region
        dat_mov = dat_mov(crop_range_x, crop_range_y, crop_range_z);
        dat_ref = dat_ref(crop_range_x, crop_range_y, crop_range_z);
        motion_current_real = motion_current_real(crop_range_x, crop_range_y, crop_range_z, :);
        
        save(sprintf("%s/Noise/%d.mat", rep_dir, cnt), ...
             "dat_mov", "dat_ref", "motion_current_real");
    end
    
    fprintf('===== Completed repeat %d/%d (elapsed time: %.2f minutes) =====\n', rep, num_repeats, toc/60);
end

fprintf('===== All data generation completed =====\n');
