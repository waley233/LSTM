function [XTrain,YTrain,XTest,YTest,mu,sig] = shujuchuli(data,numTimeStepsTrain)
dataTrain = data(1:numTimeStepsTrain+1,:);% ѵ������
dataTest = data(numTimeStepsTrain:end,:); %��֤���� 
%ѵ�����ݱ�׼������ 
mu = mean(dataTrain,'ALL');
sig = std(dataTrain,0,'ALL');
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1,:);% ѵ������ 
YTrain = dataTrainStandardized(2:end,:);% ѵ�����
%����������׼������ 
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1,:)%�������� 
YTest = dataTest(2:end,:);%������� 

XTest=XTest';
YTest=YTest';



end

