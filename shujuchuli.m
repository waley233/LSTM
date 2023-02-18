function [XTrain,YTrain,XTest,YTest,mu,sig] = shujuchuli(data,numTimeStepsTrain)
dataTrain = data(1:numTimeStepsTrain+1,:);% 训练样本
dataTest = data(numTimeStepsTrain:end,:); %验证样本 
%训练数据标准化处理 
mu = mean(dataTrain,'ALL');
sig = std(dataTrain,0,'ALL');
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1,:);% 训练输入 
YTrain = dataTrainStandardized(2:end,:);% 训练输出
%测试样本标准化处理 
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1,:)%测试输入 
YTest = dataTest(2:end,:);%测试输出 

XTest=XTest';
YTest=YTest';



end

