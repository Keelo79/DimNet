%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;             % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
h5_size_label = 320;
factor = 4;             % SR factor
downRatio = 1/factor;
sourceDataPath = './Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:3) = [];
datasetsNum = length(sourceDatasets);
idx = 0;

SavePath = ['./TrainingData_', num2str(angRes), 'x', num2str(angRes), '_', num2str(factor), 'xSR/'];
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end

%% Training data generation
for DatasetIndex = 1 : datasetsNum
    datasetsNum
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/training/'];
    folders = dir(sourceDataFolder);
    folders(1:3) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating training data of Scene_%s in Dataset %s......\t\t', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF;
        [U, V, ~, ~, ~] = size(LF);
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);  % [angRes,angRes,H,W,~]
        
        for h = 1 : floor(H/h5_size_label)
            for w = 1 : floor(W/h5_size_label)
                
                idx = idx + 1;
                idx_s = idx_s + 1;
                label = single(zeros(U,V,h5_size_label, h5_size_label));
                data = single(zeros(U,V,h5_size_label * downRatio, h5_size_label * downRatio));
                
                for u = 1 : U
                    for v = 1 : V                        
                        tempHR = squeeze(LF(u, v, (h-1)*h5_size_label+1 : h*h5_size_label, (w-1)*h5_size_label+1 : w*h5_size_label, :));
                        tempHR = rgb2ycbcr(tempHR);
                        tempHRy = squeeze(tempHR(:,:,1));
%                         x = (u-1) * patchsize + 1;
%                         y = (v-1) * patchsize + 1;
                        label(u,v, :, :) = tempHRy;
                        tempLRy = imresize(tempHRy, downRatio);
                        data(u,v, :, :) = tempLRy;
                    end
                end 

                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
                h5write(SavePath_H5, '/data', single(data));
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label));
                
            end
        end
        
        %若图片大小不为320整数倍，则在原图的右下角取一个样本，减少数据浪费
        idx = idx + 1;
                idx_s = idx_s + 1;
                label = single(zeros(U,V,h5_size_label, h5_size_label));
                data = single(zeros(U,V,h5_size_label * downRatio, h5_size_label * downRatio));
                
                for u = 1 : U
                    for v = 1 : V                        
                        tempHR = squeeze(LF(u, v, end-h5_size_label+1 : end, end-h5_size_label+1 : end, :));
                        tempHR = rgb2ycbcr(tempHR);
                        tempHRy = squeeze(tempHR(:,:,1));
%                         x = (u-1) * patchsize + 1;
%                         y = (v-1) * patchsize + 1;
                        label(u,v, :, :) = tempHRy;
                        tempLRy = imresize(tempHRy, downRatio);
                        data(u,v, :, :) = tempLRy;
                    end
                end 

                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
                h5write(SavePath_H5, '/data', single(data));
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label));
        
        fprintf([num2str(idx_s), ' training samples have been generated\n']);
    end
end

