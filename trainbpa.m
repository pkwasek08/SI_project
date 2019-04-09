clear all;
% nntwarn off
format compact;
% load dane_norm ;
load tictac ;

% P=Pn;
% T = Tn;
% [R,Q] = size(P);
[S3,Q] = size(T);
S1 = 100;
S2 = 70 ;

disp_freq=100;      
max_epoch=20000;    
err_goal=.25;     
lr=.5;  

tp = [disp_freq , max_epoch , err_goal , lr] ;
 %[W1,B1,W2,B2,W3,B3] = initff(P,S1,'tansig',S2,'tansig',T,'purelin') ;
 %[W1,B1,W2,B2,W3,B3,TE,TR] = trainbpa(W1,B1,'tansig',W2,B2,'tansig',W3,B3,'purelin',P,T,tp) ;

% A3 = simuff(P,W1,B1,'tansig',W2,B2,'tansig',W3,B3,'purelin') ;

%net = fitnet(5,'traingda');
net = feedforwardnet(10);
net = configure(net,P,T);
y1 = net(P);

%net = configure(net,P,T);    
%net=newff([-1 1], [S1, S2, S3], {'tansig','tansig','purelin'}, ...
 %  'traingda','learngdm','mse');

net.trainParam.lr=0.05;
net.trainParam.lr_inc=1.05;
net.trainParam.lr_dec=0.7;
net.trainParam.max_fail=1000;
net.trainFcn = 'traingda';

% net.divideFcn='divideind';
 net.divideParam.trainInd=1:length(P);
 net.divideParam.valInd=1:length(P);
 net.divideParam.testInd=1:length(P);
 net.trainParam.show = 100;
 net.trainParam.epochs = 10000;
 net.trainParam.goal = 0.25;
 %net.trainParam.lr = 0.01;
 net.performFcn = 'sse';

%[net,TE,TR] = trainbpa(net,P,T, tp);
[net,tr] = train(net,P,T);

y2=net(P);
plot(P,T,'o',P,y1,'x',P,y2,'*');
A3=sim(net,P);

[T' A3' (T-A3)' (abs(T-A3)>0.5)']
(1-sum(abs(T-A3)>0.5)/length(P))*100