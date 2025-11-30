%% need to save the vars in average_eva.m named "int_imp_collect" as "IntensityError.mat" and  "res_imp_collect" as "MotionError.mat" first.

clear;clc;
mat_files={'IntensityErrorr.mat','MotionError.mat'};
Data={'R','Amp','Noise'};
row_col_info.R.row_names = {"without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p",'WBI_without_pyramid'}; 
row_col_info.R.col_names = {'Rigid Score','R=5','R=6','R=7','R=8','R=9','R=10','R=11','R=12','R=13','R=14','R=15','R=16','R=17','R=18','R=19','R=20'}; % R 的列名

row_col_info.Amp.row_names = {"without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p",'WBI_without_pyramid'};
row_col_info.Amp.col_names = {'Motion Amplitude','Amp=1','Amp=2','Amp=3','Amp=4','Amp=5','Amp=6','Amp=7','Amp=8','Amp=9','Amp=10'}; % Amp 的列名

row_col_info.Noise.row_names ={"without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p",'WBI_without_pyramid'};
row_col_info.Noise.col_names = {'Noise Level','Noise=1.4^0','Noise=1.4^1','Noise=1.4^2','Noise=1.4^3','Noise=1.4^4','Noise=1.4^5','Noise=1.4^6','Noise=1.4^7','Noise=1.4^8'}; % Noise 的列名（只保留前9列）

for file_idx = 1:length(mat_files)
    mat_file = mat_files{file_idx};
    [~, name, ~] = fileparts(mat_file);
    load(mat_file);
    output_dir = fullfile(pwd, name);
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir)
    end

    for k = 1:10
        cell_data = all_int_imp(:,:,k);
        xlsx_filename = fullfile(output_dir, sprintf('rep_%d.xlsx', k));
        
        for s = 1:3
            sheetname = Data{s};
            data_to_write = cell_data{s}.';
            
            % 
            current_row_names = row_col_info.(sheetname).row_names;
            current_col_names = row_col_info.(sheetname).col_names;
            

            if strcmp(sheetname, 'Noise') && size(data_to_write, 2) > 9
                data_to_write = data_to_write(:, 1:9);
                current_col_names = current_col_names(1:10); % 只保留前9列名
            end
            

            if length(current_row_names) ~= size(data_to_write, 1)
                error('Num of rows dismatch the data！please check row "%s"', sheetname);
            end
            if length(current_col_names) ~= size(data_to_write, 2)+1
                error('Num of columns dismatch the data！please check col "%s"', sheetname);
            end
            output_data = [current_col_names; [current_row_names', num2cell(data_to_write)]];
            % write Excel
            writecell(output_data, xlsx_filename, 'Sheet', sheetname);
        end
    end
end
