function [HVI] = EvalParetoApp(CSVInput)
%=========================================================================
TimeHorizon = 5;
%-------------------------------------------------------------------------
EquipDBCSV = 'EquipDB.csv';
ClusterDBCSV = 'ClusterDB.csv';
MPDBCSV = 'MPDB.csv';
%=========================================================================

%=========================================================================
X = csvread(CSVInput);
EquipDB = csvread(EquipDBCSV);
T = csvread(ClusterDBCSV);
ClusterDB = T(:,2:size(T,2));
clear T
T = csvread(MPDBCSV);
MPDB = T(:,2:size(T,2));
clear T
%=========================================================================

%=========================================================================
ProblemData.NTransf = size(EquipDB,1);
ProblemData.NMProc = size(MPDB,1);
ProblemData.TimeHorizon = TimeHorizon;
%-------------------------------------------------------------------------
for i = 1:size(EquipDB,1)
    ProblemData.Transf(i).Id = EquipDB(i,1);
    ProblemData.Transf(i).CurrentAge = EquipDB(i,2);
    ProblemData.Transf(i).Cluster = EquipDB(i,3);
    ProblemData.Transf(i).FailCost = EquipDB(i,4);
    ProblemData.Transf(i).RiskScore = -1;
end
%-------------------------------------------------------------------------
ProblemData.ClusterPar = ClusterDB;
%-------------------------------------------------------------------------
for i = 1:size(MPDB,1)
    ProblemData.MP(i).Cost = MPDB(i,2);
    ProblemData.MP(i).AgingFactor = MPDB(i,1);
end
%=========================================================================

%=========================================================================
XminF1 = ones(1,ProblemData.NTransf);
XminF2 = ProblemData.NMProc*ones(1,ProblemData.NTransf);
FminF1 = fEvalSolution(XminF1,ProblemData);
FminF2 = fEvalSolution(XminF2,ProblemData);
%-------------------------------------------------------------------------
Fv = zeros(2,size(X,1));
for i = 1:size(X,1)
    fprintf('Evaluating Solution #%4d\n',i);
    Ft = fEvalSolution(X(i,:),ProblemData);
    Fv(1,i) = Ft(1);
    Fv(2,i) = Ft(2);
end
%=========================================================================

%=========================================================================
clc
fprintf('Solutions before filtering: %4d\n',size(Fv,2));
%=========================================================================

%=========================================================================
Fref = [FminF1 FminF2];
Utopic = min(Fref')';
Nadir = max(Fref')';
%-------------------------------------------------------------------------
[par] = fndpareto(Fv);
Fv = Fv(:,par);
%-------------------------------------------------------------------------
ToKeep = find(Fv(1,:) <= Nadir(1) & Fv(2,:) <= Nadir(2));
Fv = Fv(:,ToKeep);
%=========================================================================

%=========================================================================
fprintf('Solutions after filtering: %4d\n\n',size(Fv,2));
fprintf('Utopic: [%9.4f %9.4f]\n',Utopic(1),Utopic(2))
fprintf('Nadir : [%9.4f %9.4f]\n\n',Nadir(1),Nadir(2))
%=========================================================================

%=========================================================================
[HVI] = EvalHVI(Fv,Utopic,Nadir);
%=========================================================================

%=========================================================================
fprintf('HVI   : %8.6f\n\n',HVI);
%=========================================================================



%=========================================================================
function [HVI] = EvalHVI(front,minref,maxref)
%=========================================================================
[~,b] = sort(front(1,:));
front = front(:,b);
for i = 1:size(front,1)
    front(i,:) = (front(i,:) - minref(i)) / (maxref(i) - minref(i));
end
nadir = [1; 1];
%-------------------------------------------------------------------------
area = 0;
for i = 1:length(front)
    if i < length(front)
        area = area + ((front(1,i+1)-front(1,i))*(nadir(2,1)-front(2,i)));
    else
        area = area + ((nadir(1,1)-front(1,i))*(nadir(2,1)-front(2,i)));
    end
end
%-------------------------------------------------------------------------
HVI = area;
%=========================================================================



%=========================================================================
function [F] = fEvalSolution(XSol,ProblemData)
%=========================================================================
Fcost = 0;
Frisk = 0;
%-------------------------------------------------------------------------
for i = 1:length(XSol)
    Fcost = Fcost + ProblemData.MP(XSol(i)).Cost;
    %---------------------------------------------------------------------
    Transf = ProblemData.Transf(i);
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    eta = ProblemData.ClusterPar(Transf.Cluster,1);
    beta = ProblemData.ClusterPar(Transf.Cluster,2);
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    DeltaAge = ProblemData.TimeHorizon * ProblemData.MP(XSol(i)).AgingFactor;
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Ft = wblcdf(Transf.CurrentAge+DeltaAge,eta,beta);
    Ft0 = wblcdf(Transf.CurrentAge,eta,beta);
    Risk = (Ft - Ft0) / (1 - Ft0);
    FailCost = Risk * Transf.FailCost;
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Frisk = Frisk + FailCost;
end
%-------------------------------------------------------------------------
F = [Fcost ; Frisk];
%=========================================================================



%=========================================================================
function [par] = fndpareto(fobj)
%=========================================================================
%FNDPARETO Returns the indices of Pareto-optimal individuals in a set of
%objective function values.
%   FNDPARETO(FOB) returns a vector with the indices of Pareto-optimal
%   solutions based on the Pareto principle. In FOB each row represents one
%   objective function and each colunm represents one individual. It is
%   assumed a minimization problem.
%
%   For example:
%
%       Suposing a population with 14 individuals evaluated by two
%       functions:
%           Function 1: [1 2 2 1 1 2 3 4 4 5 4 8 6 7]
%           Function 2: [7 8 8 7 7 6 5 4 4 3 4 8 2 1]
%
%       The individuals 1, 6, 7, 8, 10, 13 and 14 are non dominated
%       individuals.
%
%       In this case, the FOB input array is:
%           [1 2 2 1 1 2 3 4 4 5 4 8 6 7;
%            7 8 8 7 7 6 5 4 4 3 4 8 2 1]
%
%       And the output is : [1 6 7 8 10 13 14]
%
%Created by Eduardo Gontijo Carrano (egcarrano@ufmg.br)
%=========================================================================
nobj = size(fobj,1);
nind = size(fobj,2);
par = [];
%-------------------------------------------------------------------------
for i = 1:nind
    lme = zeros(nobj,nind);
    leq = zeros(nobj,nind);
    for j = 1:nobj
        lme(j,:) = (fobj(j,:) <= fobj(j,i));
        leq(j,:) = (fobj(j,:) == fobj(j,i));
    end
    if isempty(find(sum(lme) == nobj & sum(leq) < nobj,1))
        lid = find(sum(lme) == nobj & sum(leq) == nobj);
        if isempty(lid)
            par = [par i];
        else
            if i <= min(lid)
                par = [par i];
            end
        end
        clear lid
    end
    clear lme leq
end
%-------------------------------------------------------------------------
[~,b] = sort(fobj(1,par));
par = par(b);
%=========================================================================
