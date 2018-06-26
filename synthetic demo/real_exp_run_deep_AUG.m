function [] = real_exp_run_deep_AUG( )
disp('real_malaria_incomplete_AUG');
exp_description = 'synthetic_data_unknown_factors';

NIter = 5;
N = 60;
MissingRate = 0.5;
LDim = 10;
TForTrain = 90;

%% Initial

IterationSet = [0.1,0.2,0.3,0.4,0.5];


METHOD_INDEX = [1,2];
for MissingMode = {'MCAR','MPS-MC','MSS-RS'} 
    time_string = regexprep(strcat(num2str(clock())), '[ .]*', '_');
    NMethod = 5;
    MAE = zeros(NMethod,length(IterationSet),NIter);
    MSE = zeros(NMethod,length(IterationSet),NIter);
    ME = zeros(NMethod,length(IterationSet),NIter);
    iCount = 0;
    
    %% Generate Synthetic Data
    file_string = strcat(['exp_data/',MissingMode{1},'_MR_',num2str(0.1*10),'_Iter_',num2str(1),'.mat']);  
    if exist(file_string,'file') == 2 
        disp( 'file exist.')
    else
        disp( 'no file exist.')
    end
    
%     str = 'n';
%     str = input('generate new data? y/n ','s');
% 
%     if str == 'y'
%         disp('generating new data.')
%         for MissingRate = (0.1:0.1:0.7)
%             for iTer = 1: NIter
%                 file_string = strcat(['exp_data/',MissingMode{1},'_MR_',num2str(MissingRate*10),'_Iter_',num2str(iTer),'.mat']);
%                 [Data ] = GenerationSynthetic(MissingMode{1},N,MissingRate,TForTrain,LDim);
%                 save(file_string,'Data');
%             end
%         end
%     end
    
    
    for MissingRate = IterationSet
        
        iCount = iCount + 1
            for iTer = 1 : NIter

                
                %% Deep Model without considering the unknown factors.
        
                file_string = strcat(['exp_data/',MissingMode{1},'_MR_',num2str(MissingRate*10),'_Iter_',num2str(iTer),'.mat']);
                load(file_string);
                
                if any(METHOD_INDEX ==1)
                    [Result1 ] = TrainningSynthetic_incomplete_deep( Data.I,Data);
                    [EVA] = Testing_incomplete_deep(Data.IForTest,Result1,Data);
                    ii = 1;
                    MAE(ii,iCount,iTer) = EVA.MAE;
                    MSE(ii,iCount,iTer) = EVA.MSE;
                    ME(ii,iCount,iTer) = EVA.ME;
                
             
                end
                
                %% Deep Model with considering the unknown factors.       
                if any(METHOD_INDEX ==2)
                    [Result1 ] = TrainningSynthetic_incomplete_deep_AUG( Data.I,Data);
                    [EVA] = Testing_incomplete_deep_AUG(Data.IForTest,Result1,Data);
                    ii = 2;
                    MAE(ii,iCount,iTer) = EVA.MAE;
                    MSE(ii,iCount,iTer) = EVA.MSE;
                    ME(ii,iCount,iTer) = EVA.ME;
                 
                end
                
                               
                save(strcat(['Real_Result/',time_string,'_Result']));
                
            end
    end
end

end

