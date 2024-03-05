function [hA,hB]=find_barrier_height(A,B,score_center,cov_cluster,mu_hat,max_value_lim)
% A,B:attractor center position 2D
% GM: the pdf of Gauss-mixed Model
%A=score_center(A_set(i)+1,:)
%B=score_center(B_set(i)+1,:)
GM = gmdistribution(score_center,cov_cluster,mu_hat);
step1=(B(1)-A(1))/100;
step2=(B(2)-A(2))/100;
%A(1:2)=[2,3];
x=A(1):step1:B(1);
y=A(2):step2:B(2);

% % 定义均值
% mu = [1 2;2 3;3 4];
% 
% % 定义协方差
% sigma = cat(3,[2 0;0 .5],[1 0;0 1],[.5 0;0 .5]);
% 
% % 定义混合成分的先验概率
% p = [0.4,0.4,0.2];
% 
% % 创建高斯混合模型
% gm = gmdistribution(mu,sigma,p);
% 
% % 生成随机样本
% rng('default');  % For reproducibility
% X = random(gm,1000);
% 
% % 绘制样本
% scatter(X(:,1),X(:,2),10,'.')
% 
% y = pdf(gm,x);

% 定义函数
f = @(x) min(-log(pdf(GM,x)),max_value_lim);

% 初始猜测值
x0 = A;
% 使用fminunc找到最小值
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
[z, fval] = fminunc(f, x0, options);
ha=fval;
% 输出结果
fprintf('The minimum value of the function is at x = %f, y = %f\n', z(1), z(2));
fprintf('The minimum value of the function is %f\n', fval);

% 初始猜测值
x0 = B;
% 使用fminunc找到最小值
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
[z1, fval1] = fminunc(f, x0, options);
hb=fval1;
% 输出结果
fprintf('The minimum value of the function is at x = %f, y = %f\n', z1(1), z1(2));
fprintf('The minimum value of the function is %f\n', fval1);
hs=f([x(1),y(1)]);
for i=2:length(x)
    if hs<f([x(i),y(i)])
        hs=f([x(i),y(i)]);
    end
end
hA=hs-ha;
hB=hs-hb;
% % 使用fmaxunc找到最小值
% options = optimoptions(@fmaxunc, 'Algorithm', 'quasi-newton');
% x0 = [2, 3];
% [x, fval] = fmaxunc(f, x0, options);
end