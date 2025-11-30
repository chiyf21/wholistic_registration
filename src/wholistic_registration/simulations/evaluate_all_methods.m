function [int_imp_collect,res_imp_collect]=evaluate_all_methods(optionPath,rawDataPath,wbiPath,BSplinePath,normcorrePath,suite2pPath,DemonsPath,wbi2Path,plot)
    %% file path
    % optionPath : the path of option
    % rootfilePath : the path of the registrated data
    load(optionPath,"lst_R","lst_Amp","lst_Noise");
    varNameLst=["R" "Amp" "Noise"];

    res_imp_collect=cell(length(varNameLst),1);
    int_imp_collect=cell(length(varNameLst),1);
    
    res_all_collect=cell(length(varNameLst),1);
    int_all_collect=cell(length(varNameLst),1);
    
    edgeSpace=20;
    x_idx=1+edgeSpace:256-edgeSpace;
    y_idx=1+edgeSpace:256-edgeSpace;
    
    %%
    for cnt=1:length(varNameLst)
        varName=varNameLst(cnt);
        disp(varName)
    
        filePath=rawDataPath+varName+"/";
        
        
        N=length(dir(filePath))-2;
        methodCnt=8;
       
        res_imp=nan(N,methodCnt); % result of textured pixel with motion
        int_imp=nan(N,methodCnt);
        res_all=nan(N,methodCnt); % result of textured pixel with motion
        int_all=nan(N,methodCnt);
        disp(size(res_imp))
        for filecnt=1:N
            disp(filecnt)
    
            % Evaluate raw data
            load(rawDataPath+varName+"/"+num2str(filecnt),"motion_current_real","dat_ref","dat_mov");
            motion_current_real=motion_current_real(x_idx,y_idx,:,:);
            dat_mov_raw=dat_mov;
            dat_ref_raw=dat_ref;

            dat_ref=dat_ref(x_idx,y_idx,:);
            dat_mov=dat_mov(x_idx,y_idx,:);
            disp(size(motion_current_real));

            validMap=true(size(motion_current_real(:,:,:,1:2)));
            validMap_int=validMap(:,:,:,1)|validMap(:,:,:,2);

            temp=motion_current_real.^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,1)=mean(temp(validMap),"all");
            res_all(filecnt,1)=mean(temp,"all");
            temp=(dat_ref-dat_mov).^2;
            int_imp(filecnt,1)=mean(temp(validMap_int),"all");
            int_all(filecnt,1)=mean(temp,"all");
    
            % Evaluate ground truth motion field
            temp=(motion_current_real.*0).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,2)=mean(temp(validMap),"all");
            res_all(filecnt,2)=mean(temp,"all");
            dat_cor=correctMotion_Wei_v2(dat_mov,motion_current_real);
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,2)=mean(temp(validMap_int),"all");
            int_all(filecnt,2)=mean(temp,"all");
    
            % Evaluate WBI motion fild
            motionPath=wbiPath+varName+"/";
            load(motionPath+filecnt+".mat","motion","dat_cor");
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            disp(size(motion))
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,3)=mean(temp(validMap),"all");
            res_all(filecnt,3)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,3)=mean(temp(validMap_int),"all");
            int_all(filecnt,3)=mean(temp,"all");
    
    
            % Evaluate Bspline1 motion field
            motionPath=BSplinePath+varName+"/";
            motion=h5read(motionPath+filecnt+'/'+'result.h5','/displacement_field');
            motion=permute(motion,[2,3,4,1]);
            motion(:,:,:,3)=motion(:,:,:,3)/27;
            dat_cor=correctMotion_Wei_v2(dat_mov_raw,motion);
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,4)=mean(temp(validMap),"all");
            res_all(filecnt,4)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,4)=mean(temp(validMap_int),"all");
            int_all(filecnt,4)=mean(temp,"all");


            %Evaluate Demons motion field
            motionPath=DemonsPath+varName+"/";
            motion=h5read(motionPath+filecnt+'/'+'result.h5','/displacement_field');
            motion=permute(motion,[2,3,4,1]);
            motion(:,:,:,3)=motion(:,:,:,3)/27;
            dat_cor=correctMotion_Wei_v2(dat_mov_raw,motion);
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,5)=mean(temp(validMap),"all");
            res_all(filecnt,5)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,5)=mean(temp(validMap_int),"all");
            int_all(filecnt,5)=mean(temp,"all");

            % Evaluate NoRMCorre motion field
            motionPath=normcorrePath+"/"+varName+"/";
            load(motionPath+"corrected_"+filecnt+".mat","motion","dat_cor");
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,6)=mean(temp(validMap),"all");
            res_all(filecnt,6)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,6)=mean(temp(validMap_int),"all");
            int_all(filecnt,6)=mean(temp,"all");
    
            % Evaluate suite2p motion field
            motionPath=suite2pPath+"/"+varName+"/";
            load(motionPath+"corrected_"+filecnt+".mat","motion","dat_cor");
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,7)=mean(temp(validMap),"all");
            res_all(filecnt,7)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,7)=mean(temp(validMap_int),"all");
            int_all(filecnt,7)=mean(temp,"all");
   
            % Evaluate WBI motion fild
            motionPath=wbi2Path+varName+"/";
            load(motionPath+filecnt+".mat","motion","dat_cor");
            motion=motion(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:,:);
            disp(size(motion))
            dat_cor=dat_cor(1+edgeSpace:end-edgeSpace,1+edgeSpace:end-edgeSpace,:);
            temp=(motion_current_real-motion).^2;
            temp=temp(:,:,:,1:2);
            res_imp(filecnt,8)=mean(temp(validMap),"all");
            res_all(filecnt,8)=mean(temp,"all");
            temp=(dat_ref-dat_cor).^2;
            int_imp(filecnt,8)=mean(temp(validMap_int),"all");
            int_all(filecnt,8)=mean(temp,"all");
   
        end
        
        res_imp_collect{cnt}=res_imp;
        int_imp_collect{cnt}=int_imp;
    
        res_all_collect{cnt}=res_all;
        int_all_collect{cnt}=int_all;
    end

    
    
    int_imp_collect{1}(:,2)=[];
    int_imp_collect{2}(:,2)=[];
    int_imp_collect{3}(:,2)=[];
    
    
    res_imp_collect{1}(:,2)=[];
    res_imp_collect{2}(:,2)=[];
    res_imp_collect{3}(:,2)=[];
    
    if plot==true
        %%
        fig = figure();
        set(fig, 'Position', [10 10 1000 1000]);
        disp(size(lst_R))
        disp(size(res_imp_collect{1}))
        subplot(3, 2, 1);
        plot(lst_R, res_imp_collect{1}, LineWidth= 3);
        title("Distance MSE change with rigid level");
        xlabel("rigid score");
        ylabel("MSE");
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
        
        subplot(3, 2, 3);
        plot(lst_Amp, res_imp_collect{2},LineWidth= 3);
        title("Distance MSE change with motion amplitude");
        xlabel("motion amplitude");
        ylabel("MSE");
        axis([lst_Amp(1), lst_Amp(end), -inf, inf]);
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
        
        subplot(3, 2, 5);
        semilogx(lst_Noise, res_imp_collect{3}, LineWidth= 3);
        title("Distance MSE change with noise level");
        xlabel("noise level");
        ylabel("MSE");
        axis([lst_Noise(1), lst_Noise(end), -inf, inf]);
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
        
        subplot(3, 2, 2);
        plot(lst_R, int_imp_collect{1}, LineWidth= 3);
        title("Intensity MSE change with rigid level");
        xlabel("rigid score");
        ylabel("MSE");
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
        
        subplot(3, 2, 4);
        plot(lst_Amp, int_imp_collect{2}, LineWidth= 3);
        title("Intensity MSE change with motion amplitude");
        xlabel("motion amplitude");
        ylabel("MSE");
        axis([lst_Amp(1), lst_Amp(end), -inf, inf]);
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
        
        subplot(3, 2, 6);
        semilogx(lst_Noise, int_imp_collect{3}, LineWidth= 3);
        title("Intensity MSE change with noise level");
        xlabel("noise level");
        ylabel("MSE");
        axis([lst_Noise(1), lst_Noise(end), -inf, inf]);
        set(gca, 'YScale', 'log');
        legend("without correction", "WBI");
    end
end
