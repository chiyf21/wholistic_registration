clear;clc;
all_int_imp = [];
all_res_imp = [];

selected_reps=[1,2,3,4,5,6,7,8,9,10];

% Loop through all repetitions
load("/home/cyf/wbi/Virginia/result/opt.mat")


for rep_num = selected_reps
    % Format the repetition number with leading zero (01, 02, ..., 10)
    rep_str = sprintf('rep%02d', rep_num);
    
    % Construct paths for this repetition
    opt_path = '/home/cyf/wbi/Virginia/result/opt.mat';
    sim_path = fullfile('/home/cyf/wbi/Virginia/final_simulation_0907/', rep_str, '/');
    eval_path = fullfile('/home/cyf/wbi/evaluation_result_0907/wbi/', rep_str, '/');

    rawDataPath=fullfile('/home/cyf/wbi/Virginia/simulation/final_simulation_0907/',rep_str,'/');
    wbiPath=fullfile('/home/cyf/wbi/evaluation_result_0907/wbi/',rep_str,'/');
    BSplinePath=fullfile('/home/cyf/wbi/evaluation_result_0907/BSpline/',rep_str,'/');
    normcorrePath=fullfile('/home/cyf/wbi/evaluation_result_0907/normcorre/',rep_str,'/');
    suite2pPath=fullfile('/home/cyf/wbi/evaluation_result_0907/suite2p/',rep_str,'/');
    DemonsPath=fullfile('/home/cyf/wbi/evaluation_result_0907/Demon/',rep_str,'/');
    wbi2Path=fullfile('/home/cyf/wbi/evaluation_result_0907/wbi_nopyramid/',rep_str,'/');

    % Run evaluation for this repetition
    [int_imp_collect, res_imp_collect] = evaluate_all_methods(opt_path, rawDataPath,wbiPath,BSplinePath,normcorrePath,suite2pPath,DemonsPath,wbi2Path,false);

    all_int_imp = cat(3, all_int_imp, int_imp_collect);  % Assuming results are 2D matrices
    all_res_imp = cat(3, all_res_imp, res_imp_collect);
    
    % Optional: Display progress
    fprintf('Completed processing %s\n', rep_str);
end


mean_int_imp = cell(3, 1);
mean_res_imp = cell(3, 1);
std_int_imp = cell(3, 1);
std_res_imp = cell(3, 1);
for i = 1:3
    int_data_list = all_int_imp(i,1,:);
    res_data_list = all_res_imp(i,1,:);
    temp_int = cat(3, int_data_list{:});  % Nx × Nmethods × Nrep
    temp_res = cat(3, res_data_list{:});
    std_int_imp{i} = std(temp_int, 0, 3, 'omitnan');
    std_res_imp{i} = std(temp_res, 0, 3, 'omitnan');
end


nReps = numel(selected_reps);
alpha = 0.05;
tval = tinv(1 - alpha/2, nReps - 1); 


mean_int_imp = cell(3, 1);
mean_res_imp = cell(3, 1);
ci_int_imp = cell(3, 1);
ci_res_imp = cell(3, 1);



for i = 1:3

    int_data_list = all_int_imp(i,1,:);
    res_data_list = all_res_imp(i,1,:);
    

    temp_int = cat(3, int_data_list{:});  
    temp_res = cat(3, res_data_list{:});
    

    mean_int_imp{i} = mean(temp_int, 3,'omitnan');  
    mean_res_imp{i} = mean(temp_res, 3,'omitnan');


    std_int = std(temp_int, 0, 3, 'omitnan');
    std_res = std(temp_res, 0, 3, 'omitnan');


    ci_int_imp{i} = tval * std_int / sqrt(nReps);
    ci_res_imp{i} = tval * std_res / sqrt(nReps);

end


mean_int_imp = reshape(mean_int_imp, [3, 1]);
mean_res_imp = reshape(mean_res_imp, [3, 1]);

res_imp_collect=mean_res_imp;
int_imp_collect=mean_int_imp;
fig= figure();
set(fig, 'Position', [10 10 1000 1000]);
disp(size(lst_R))
disp(size(res_imp_collect{1}))



%% plot result

colors = lines(7);  

subplot(3, 2, 1); hold on;
for k = 1: 7
    errorbar(lst_R, res_imp_collect{1}(:,k), std_res_imp{1}(:,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Distance MSE change with rigid level");
xlabel("rigid score"); ylabel("MSE"); set(gca, 'YScale', 'log');
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");

subplot(3, 2, 3); hold on;
for k = 1:7
    errorbar(lst_Amp, res_imp_collect{2}(:,k), std_res_imp{2}(:,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Distance MSE change with motion amplitude");
xlabel("motion amplitude"); ylabel("MSE"); set(gca, 'YScale', 'log');
axis([lst_Amp(1), lst_Amp(end), -inf, inf]);
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");

subplot(3, 2, 5); hold on;
for k = 1:7
    errorbar(lst_Noise(1:9), res_imp_collect{3}(1:9,k), std_res_imp{3}(1:9,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Distance MSE change with noise level");
xlabel("noise level"); ylabel("MSE"); set(gca, 'YScale', 'log');
axis([lst_Noise(1), lst_Noise(end-1), -inf, inf]);
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");

subplot(3, 2, 2); hold on;
for k = 1:7
    errorbar(lst_R, int_imp_collect{1}(:,k), std_int_imp{1}(:,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Intensity MSE change with rigid level");
xlabel("rigid score"); ylabel("MSE"); set(gca, 'YScale', 'log');
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");

subplot(3, 2, 4); hold on;
for k = 1:7
    errorbar(lst_Amp, int_imp_collect{2}(:,k), std_int_imp{2}(:,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Intensity MSE change with motion amplitude");
xlabel("motion amplitude"); ylabel("MSE"); set(gca, 'YScale', 'log');
axis([lst_Amp(1), lst_Amp(end), -inf, inf]);
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");

subplot(3, 2, 6); hold on;
for k = 1:7
    errorbar(lst_Noise(1:9), int_imp_collect{3}(1:9,k), std_int_imp{3}(1:9,k), ...
        'o-', 'LineWidth', 2, 'Color', colors(k,:), 'CapSize', 4);
end
title("Intensity MSE change with noise level");
xlabel("noise level"); ylabel("MSE"); set(gca, 'YScale', 'log');
axis([lst_Noise(1), lst_Noise(end-1), -inf, inf]);
legend("without correction", "WBI","BSpline","Demons","NoRMCorre","suite2p","WBI without pyramid");
