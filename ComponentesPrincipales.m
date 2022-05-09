clear all, close all, clc
tic
load ovariancancer;  % Datos de cancer de ovario 121 cancer / 95 sin cancer
X=obs'; % mxn = 4000x216 = caracteristicas x personas 
Xmed=(X*ones(216,1))/216; % Valor medio de [mxn][nx1] = [mx1]
Xaux = (X-Xmed);
covar = (Xaux*Xaux')/(216-1); %Covarianza formula 
covar_ = cov(X'); % Covarianza funcion matlab

if(covar - covar_ <= 1e-4) disp('OK') 
else disp('BAD')
end

%% Singular Value descomposition (SVD)
tic
[U,S,V] = svd(covar);
toc
%%  PCA al 95%
for i=1:10
    H(:,i) = [i;100 * trace(S(1:i,1:i))/trace(S)];
    normas(:,i) = norm(V(:,i));
end
H(:,1:10)

%% Norma de V = 1
normas(:,1:10)
%% Graficas de todos los autovaloes y los componentes principales 
figure (1)
subplot(1,2,1)
semilogy(diag(S),'b-o','LineWidth',1)
set(gca,'FontSize',13), axis tight, grid on
subplot(1,2,2)
semilogx(cumsum(diag(S))./sum(diag(S)),'k-o','LineWidth',1.5)
set(gca,'FontSize',13), axis tight, grid on
set(gcf,'Position',[100 100 600 250])

%% Grafica 3D con proyeccion a 3 autovectores
Xred = V(:,1:3)'*X;
size(Xred)
%%
figure (2), hold on
for i=1:size(X,2) 
    if(grp{i}=='Cancer')
        plot3(Xred(1,i),Xred(2,i),Xred(3,i),'rx','LineWidth',2);
    else
        plot3(Xred(1,i),Xred(2,i),Xred(3,i),'bo','LineWidth',2);
    end
end

toc
%plot(dt)
view(85,25), grid on, set(gca,'FontSize',13)


