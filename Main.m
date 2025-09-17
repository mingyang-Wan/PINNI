function ExpMechMain()
%****************************************************************************
%  ExpMech-Main.m
%  For mechanics problem with Explicit algorithm
%****************************************************************************
%****************************************************************************
% Variables  specification
% nof = Number of Freedom of each node 
% nod = Number of Node of each element
% nom = Number of Dimension  
% neq = Number of Equations
%****************************************************************************

%##############################################################################################
%                  Define arrays, matrix and variables
%
nom = 2; nof = 2;
% Open files 
fileID_in = fopen('Pre_in.txt', 'r');
fileID_xy = fopen('Pre_xyijmbc.txt', 'r');
fileID_idf = fopen('Post_idf.txt', 'r');
fileID_forcen = fopen('Post_forcen.txt', 'w');
fileID_disp = fopen('Post_disp.txt', 'w');
fileID_times = fopen('Post_times.txt', 'w');
fileID_velocity = fopen('Post_velocity.txt', 'w');
fileID_nsts = fopen('Post_nsts.txt', 'w');
fileID_straineig = fopen('Post_straineig.txt', 'w');

%##############################################################################################
%                                    allocate discrete data
temp = fscanf(fileID_in, '%f');
ng = temp(1);
ne = temp(2);
nject = temp(3);
swd = temp(4);
nb = temp(5);
nbg = temp(6);
nf = temp(7);
nfg = temp(8);
np = temp(9);
npg = temp(10);
nstep = temp(11);
insp = temp(12);
thi = temp(13);
dtime0 = temp(14);

% Allocate arrays for reading data
xy = zeros(2*ng, 1);
ijm = zeros(3*ne, 1);
idf = zeros(ne, 1);
nwd = zeros(nject, 1);
wd = zeros(swd, 1);
mb = zeros(2*nb, 1);
zb = zeros(nb, 1);
mf = zeros(2*nf, 1);
zf = zeros(nf, 1);
mp = zeros(2*np, 1);
zp = zeros(np, 1);
loadt = zeros(nject*2*(nstep+1), 1);
matpara = zeros(50, 1);

% Read all data from the second file
temp = fscanf(fileID_xy, '%f');
start_idx = 1;

% Fill arrays with data from file
xy = temp(start_idx:start_idx+2*ng-1);
start_idx = start_idx + 2*ng;

ijm = temp(start_idx:start_idx+3*ne-1);
start_idx = start_idx + 3*ne;

idf = temp(start_idx:start_idx+ne-1);
start_idx = start_idx + ne;

nwd = temp(start_idx:start_idx+nject-1);
start_idx = start_idx + nject;

wd = temp(start_idx:start_idx+swd-1);
start_idx = start_idx + swd;

mb = temp(start_idx:start_idx+2*nb-1);
start_idx = start_idx + 2*nb;

zb = temp(start_idx:start_idx+nb-1);
start_idx = start_idx + nb;

mf = temp(start_idx:start_idx+2*nf-1);
start_idx = start_idx + 2*nf;

zf = temp(start_idx:start_idx+nf-1);
start_idx = start_idx + nf;

mp = temp(start_idx:start_idx+2*np-1);
start_idx = start_idx + 2*np;

zp = temp(start_idx:start_idx+np-1);
start_idx = start_idx + np;

loadt = temp(start_idx:start_idx+nject*2*(nstep+1)-1);
start_idx = start_idx + nject*2*(nstep+1);

matpara = temp(start_idx:start_idx+50-1);

%##############################################################################################
%                       allocate mechanical arrays
neq = nof*ng;
u0 = zeros(neq, 1);
u1 = zeros(neq, 1);
force = zeros(neq, 1);
velo = zeros(neq, 1);
exf = zeros(neq, 1);
mass = zeros(neq, 1);
ldt = zeros(4*nject, 1);
zb1 = zeros(nb, 1);
zf1 = zeros(nf, 1);
zp1 = zeros(np, 1);
stna = zeros(ne, 1);
mass = mass_node(ng, ne, xy, ijm, matpara(5));
%##############################################################################################

%##############################################################################################
%       Explicit algorithm
tim = 0.0; 
kkk = 0; 
ostep = 0;
exf = zeros(neq, 1);

for ii = 1:nstep
    fprintf('\nMainstep ii = %d\n', ii);
    [bstep, dtime, ldt] = Loading_stepV1(ii, nject, nstep, loadt, dtime0);
    
    for jj = 1:bstep
        % BC at t1
        [tim, zb1, zf1, zp1] = Loading_subtime_stepV1(jj, bstep, dtime, ldt, nject, nwd, nb, nbg, zb, nf, nfg, zf, np, npg, zp, tim);
        
        % for nodal displacement u1
        u1 = -0.5*dtime*force + 1.0/dtime*mass.*u0 + mass.*velo + 0.5*dtime*exf;
        u1 = dtime*u1./mass;
        u1 = restbound(nof, ng, nb, nbg, mb, zb1, u1);
        
        % for external force and nodal force at t1
        exf = zeros(neq, 1);
        exf = exf_bc_2D(ng, xy, nf, mf, zf1, np, mp, zp1, exf);
        [force, stna] = nodal_force(ng, ne, xy, ijm, idf, matpara, u1, force, stna);
        
        % for nodal velocity at t1
        velo = -0.5*dtime*force + 1.0/dtime*mass.*(u1-u0) + 0.5*dtime*exf;
        velo = velo./mass;
        
        % update
        u0 = u1;
        tim = tim + dtime;
        
        % output data
        kkk = kkk + 1;
        if mod(kkk, insp) == 0
            Output_data(ng, ne, tim, u0, velo, force, stna, fileID_disp, fileID_velocity, fileID_forcen, fileID_times, fileID_straineig);
            ostep = ostep + 1;
            fprintf('Outstep = %d\n', ostep);
        end
    end
end

% Close files
fclose(fileID_in);
fclose(fileID_xy);
fclose(fileID_idf);
fclose(fileID_forcen);
fclose(fileID_disp);
fclose(fileID_times);
fclose(fileID_velocity);
fclose(fileID_nsts);
fclose(fileID_straineig);

fprintf('\nComputation complete\n');
end

%******************************************************************************
function Output_data(ng, ne, tim, u, v, force, stna, fileID_disp, fileID_velocity, fileID_forcen, fileID_times, fileID_straineig)
%==============================================================================
%   Output data
%==============================================================================
% Output nodal displacement
fprintf(fileID_disp, '%f ', single(u));
fprintf(fileID_disp, '\n');

% Output nodal velocity
fprintf(fileID_velocity, '%f ', single(v));
fprintf(fileID_velocity, '\n');

% Output nodal force
fprintf(fileID_forcen, '%f ', single(force));
fprintf(fileID_forcen, '\n');

fprintf(fileID_times, '%f\n', tim);

fprintf(fileID_straineig, '%f ', single(stna));
fprintf(fileID_straineig, '\n');
end

%******************************************************************************
function mass = mass_node(ng, ne, xy, ijm, dens)
%==============================================================================
% Lumped nodal mass array for triangular mesh
% Input: 
%           ng = node number 
%           ne = element number
%     xy(2*ng) = node coordinates
%    ijm(3*ne) = element node
%         dens = solid density
% Output:
%     mass(2*ng) = nodal mass  
%==============================================================================
mass = zeros(2*ng, 1); % Initialization

for i = 1:ne
    ijme = zeros(3, 1);
    exy = zeros(6, 1);
    
    for j = 1:3
        ik = ijm(3*(i-1)+j);
        ijme(j) = ik;
        exy(2*j-1) = xy(2*ik-1);
        exy(2*j) = xy(2*ik);
    end
    
    % Triangular area
    vol = exy(1)*exy(4)-exy(2)*exy(3) + exy(3)*exy(6)-exy(4)*exy(5) + exy(5)*exy(2)-exy(6)*exy(1);
    vol = 0.5*vol; % triangular area
    
    as = vol*dens/3.0;
    for j = 1:3
        ik = 2*ijme(j);
        mass(ik-1) = mass(ik-1) + as;
        mass(ik) = mass(ik) + as;
    end
end
end

%******************************************************************************
function [bstep, dtime, ldt] = Loading_stepV1(jj, nject, nstep, loadt, dtime0)
%==============================================================================
% Loading step 
%     Compared with Loading_Step, this version deal with multiple injection point problem
% Input: 
%          jj = loading step
%       nject = number of injection points
%       nstep = total time step
%       loadt(2*nject*(nstep+1)) = loading path in different injection points, 
%               loadt(1,:)=loading time, loadt(2,:)=loading value
%               loadt = [injection-1| injection-2|....|injection-n]
%       dtime0 = time interval 
% Output:
%        bstep = total subtime step number
%        dtime = time interval 
%        ldt(4*nject) = load step value at different injection point
%                ldt(:,1)=[t0,p0,t1,p1]
%==============================================================================
ldt = zeros(4*nject, 1);

% loading time and value at t0 and t1
for io = 1:nject
    ini = (io-1)*(nstep+1);
    ldt(4*(io-1)+1) = loadt(2*ini+2*jj-1);   % t0
    ldt(4*(io-1)+2) = loadt(2*ini+2*jj);     % load value at t0
    ldt(4*(io-1)+3) = loadt(2*ini+2*(jj+1)-1); % t1
    ldt(4*(io-1)+4) = loadt(2*ini+2*(jj+1));   % load value at t1
end

% for subtime step number
dt = ldt(3) - ldt(1); % t1-t0
bstep = ceil(dt/dtime0); % total subtime step number
dtime = dt/bstep; % time interval
end

%******************************************************************************
function [tim, zb1, zf1, zp1] = Loading_subtime_stepV1(kk, bstep, dtime, ldt, nject, nwd, nb, nbg, zb, nf, nfg, zf, np, npg, zp, tim)
%==============================================================================
% Sub Load step BC
%   Compared with the original one, this version deals with multiple injection points
% Input: 
%                      kk = loading step
%                   bstep = total time step
%                   nject = number of injection points
%              nwd(nject) = node number at each injection point
%            ldt(4*nject) = load step value at different injection point,ldt(:,1)=[t0,p0,t1,p1]
%           nb,nbg,zb(nb) = restriction, variable BC number and BC value
%           nf,nfg,zf(nf) = concentrated, variable force BC number and BC value
%           np,npg,zp(np) = distributed,variable pressure BC number and BC value
% Output:
%            tim = total time
%            zb1(nb) = restriction BC value at t1
%            zp1(np) = distributed pressure value at t1
%            zf1(nf) = concentrated force BC value at t1
%==============================================================================
% Initialization
zb1 = zb;
zf1 = zf;
zp1 = zp;

cof = kk/bstep;
cof0 = (kk-1)/bstep;
ini = 0;

for i = 1:nject
    snwd = nwd(i); % number of injection nodes at i-th injection point
    tim = ldt(4*(i-1)+1) + cof*(ldt(4*(i-1)+3)-ldt(4*(i-1)+1)); % time
    ldv = ldt(4*(i-1)+2) + cof*(ldt(4*(i-1)+4)-ldt(4*(i-1)+2)); % loading value
    ldv0 = ldt(4*(i-1)+2) + cof0*(ldt(4*(i-1)+4)-ldt(4*(i-1)+2));
    
    for j = 1:snwd
        ini = ini + 1;
        % if pressure loading
        if npg > 0
            zp1(np-npg+ini) = ldv;
        end
        % if displacement loading
        if nbg > 0
            zb1(nb-nbg+ini) = ldv;
        end
        % if force loading
        if nfg > 0
            zf1(nf-nfg+ini) = ldv;
        end
    end
end
end

%******************************************************************************
function u1 = restbound(nof, ng, nb, nbg, mb, zb1, u1)
%==================================================================================================
% Restriction boundary condition introduction
%==================================================================================================
for i = 1:nb
    ik = nof*mb(2*i-1) - mb(2*i);
    u1(ik) = zb1(i);
end
end

%******************************************************************************
function exf = exf_bc_2D(ng, xy, nf, mf, zf1, np, mp, zp1, exf)
%====================================================================================
%   External nodal force contributed by concentrated force and pressure on Boundary 
% Input:
%           ng = node number 
%     xy(2*ng) = node coordinates
%           np,mp(2*np),zp1(np) = pressure BC number, infor, value at t1
%           nf,mf(2*nf),zf1(nf) = concentrated force BC number, infor, value at t1
% Output:
%       exf(2*ng) = external force
%====================================================================================
% external force/flux by force/flux boundary conditions
for i = 1:nf
    ik = 2*mf(2*i-1)-mf(2*i);
    exf(ik) = exf(ik) + zf1(i); % force boundary
end

% external force by pressure BC
for i = 1:np
    n1 = mp(2*i-1);
    n2 = mp(2*i);
    dx = xy(2*n2-1)-xy(2*n1-1);
    dy = xy(2*n2)-xy(2*n1);
    l = sqrt(dx*dx + dy*dy); % pressure distribution length
    n = [dy/l; -dx/l]; % outwards normal direction relative to the third node of triangular element
    f1 = -0.5*l*zp1(i)*n; % each nodal force at t1
    exf(2*n1-1) = exf(2*n1-1) + f1(1);
    exf(2*n1) = exf(2*n1) + f1(2);
    exf(2*n2-1) = exf(2*n2-1) + f1(1);
    exf(2*n2) = exf(2*n2) + f1(2);
end
end

%******************************************************************************
function [f, stna] = nodal_force(ng, ne, xy, ijm, idf, mat, u, f, stna)
%====================================================================================
%   Nodal force  
% Input:
%           ng,ne = node and element number 
%           xy(2*ng) = node coordinates
%           ijm(3*ne) = element node
%           mat(50) = material parameters
%           u(2*ng) = nodal disp velocity at t1
%           idf(ne) = element state
% In-Output:
%           f(2*ng) = nodal force
%           stna(ne) = nodal force
%====================================================================================
f = zeros(2*ng, 1); % Initialization

for io = 1:ne % triangular element
    if idf(io) == 0
        % element infor
        ijme = zeros(3, 1);
        exy = zeros(6, 1);
        eu = zeros(6, 1);
        
        for i = 1:3
            ik = ijm(3*(io-1)+i);
            ijme(i) = ik;
            exy(2*i-1) = xy(2*ik-1); % node coord.
            exy(2*i) = xy(2*ik);
            eu(2*i-1) = u(2*ik-1); % nodal disp
            eu(2*i) = u(2*ik);
        end
        
        % nodal force
        [vol, b] = Bmatrix_geom_tri(exy); % Geometry parameters of triangular element
        stnv = b * eu; % strain vector
        
        % check whether this element is failed
        stn1 = stnv(1)-stnv(2);
        stn1 = 0.5*(stnv(1)+stnv(2) + sqrt(stn1*stn1 + stnv(3)*stnv(3)));
        stna(io) = stn1; % First principal strain value
        
        % element stress
        % fprintf("called\n")
        stsv = stress(stnv, mat(1), mat(2), mat(3), mat(4)); % stress vector
        
        ef = vol * (b' * stsv);
        
        % compile
        for i = 1:3
            ik = 2*ijme(i);
            f(ik-1) = f(ik-1) + ef(2*i-1);
            f(ik) = f(ik) + ef(2*i);
        end
    end
end
end

%******************************************************************************
function [vol, b] = Bmatrix_geom_tri(exy)
%==================================================================================================
%   B-matrix of 3-node triangular element, displacement-strain transform matrix
%      Input:
%           exy(6) = nodal coord
%     Output:
%           b(3,6) = b-matrix
%              vol = volume of element
%==================================================================================================
b = zeros(3, 6); % Initialization

% for geometry parameters
bt = zeros(7, 1);
bt(1) = exy(4)-exy(6);
bt(2) = exy(6)-exy(2);
bt(3) = exy(2)-exy(4);
bt(4) = exy(5)-exy(3);
bt(5) = exy(1)-exy(5);
bt(6) = exy(3)-exy(1);
bt(7) = 0.5*bt(1)*bt(5)-0.5*bt(4)*bt(2);
vol = bt(7);

% comput b matrix
as = 0.5/bt(7); % 1/2/area of triangular element
for i = 1:3
    ik = 2*i;
    b(1, ik-1) = bt(i)*as;   b(1, ik) = 0.0;
    b(2, ik-1) = 0.0;        b(2, ik) = bt(i+3)*as;
    b(3, ik-1) = bt(i+3)*as; b(3, ik) = bt(i)*as;
end
end

%******************************************************************************
function stsv = stress(stnv, ym, pr, stnt, stnb)
%==================================================================================================
%  Constitutive model of Complete AVIB model (Zhang and Gao, 2012)
%  Input:
%     stnv(3) = strain vector
%          ym = Young's modulus
%          pr = Poisson ratio
%        stnt = tensile strain strength
%        stnb = compressive strain strength
%  Output:
%     stsv(3) = stress vector
%==================================================================================================
% Gauss points for integration
gx = [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142];
gh = [0.1713244924, 0.3607615730, 0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924];
cf = [0.39269908, 1.178097245, 1.963495085, 2.74889357]; % pi/8, 3pi/8, 5pi/8, 7pi/8

% Initialize
d = zeros(3, 3);
stsv = zeros(3, 1);

% nonlinear model
lama = ym/(pi*(1.0-pr));
lamb = ym*(1.0-3.0*pr)/(pi*(1.0-pr)*(1.0+pr));
q = (1.0-3.0*pr)/(2.0*(1.0+pr));

c = zeros(4, 1);
for j1 = 1:4
    for i1 = 1:6
        thi = 0.39269908*gx(i1)+cf(j1);
        x1 = sin(thi); x2 = cos(thi); % micro bond direction vector x=(x1,x2)
        y1 = stnv(1)*x1+0.5*stnv(3)*x2; % y=estn*x (y1,y2)
        y2 = 0.5*stnv(3)*x1+stnv(2)*x2;
        dn = x1*y1+x2*y2; % bond stretch
        rt = y1*y1+y2*y2-dn*dn; % bond rotation^2
        stc = stnt; % tension
        if dn < 0.0
            stc = -stnb; % compression
        end
        c1 = exp(-dn/stc);
        c2 = exp(-rt/(stc*stc));
        fa = lama*dn*c1*(1.0-q+q*c2);
        fb = lamb*c1*(1.0+dn/stc)*c2;
        a = zeros(4, 1);
        a(1) = fa*x1*x1 + fb*(y1*x1 - dn*x1*x1); % s11
        a(2) = fa*x1*x2 + fb*(y1*x2 - dn*x1*x2); % s12
        a(3) = fa*x2*x1 + fb*(y2*x1 - dn*x2*x1); % s21
        a(4) = fa*x2*x2 + fb*(y2*x2 - dn*x2*x2); % s22
        c = c + a*0.39269908*gh(i1);
    end
end

stsv(1) = 2.0*c(1); % 2*pi
stsv(2) = 2.0*c(4);
stsv(3) = c(2)+c(3);
end

%******************************************************************************
function stsv = stress0(stnv, ym, pr, stnt, stnb)
%==================================================================================================
%  Constitutive model of Complete AVIB model (Zhang and Gao, 2012)
%  Input:
%     stnv(3) = strain vector
%          ym = Young's modulus
%          pr = Poisson ratio
%        stnt = tensile strain strength
%        stnb = compressive strain strength
%  Output:
%     stsv(3) = stress vector
%==================================================================================================
% Gauss points for integration
gx = [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142];
gh = [0.1713244924, 0.3607615730, 0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924];
cf = [0.39269908, 1.178097245, 1.963495085, 2.74889357]; % pi/8, 3pi/8, 5pi/8, 7pi/8

% Initialize
d = zeros(3, 3);
stsv = zeros(3, 1);

% nonlinear model
lama = ym/(pi*(1.0-pr));
lamb = ym*(1.0-3.0*pr)/(pi*(1.0-pr)*(1.0+pr));

c = zeros(4, 1);
for j1 = 1:4
    for i1 = 1:6
        thi = 0.39269908*gx(i1)+cf(j1);
        x1 = sin(thi); x2 = cos(thi); % micro bond direction vector x=(x1,x2)
        y1 = stnv(1)*x1+0.5*stnv(3)*x2; % y=estn*x
        y2 = 0.5*stnv(3)*x1+stnv(2)*x2;
        dn = x1*y1+x2*y2; % bond stretch
        rt = y1*y1+y2*y2-dn*dn; % bond rotation^2
        
        % first derivative of bond potential
        fa = lama*dn;
        if dn > 0.0
            fa = lama*dn*exp(-dn/stnt);
        end
        fb = lamb*exp(-rt/stnb/stnb);
        
        a = zeros(4, 1);
        a(1) = fa*x1*x1 + fb*(y1*x1 - dn*x1*x1); % s11
        a(2) = fa*x1*x2 + fb*(y1*x2 - dn*x1*x2); % s12
        a(3) = fa*x2*x1 + fb*(y2*x1 - dn*x2*x1); % s21
        a(4) = fa*x2*x2 + fb*(y2*x2 - dn*x2*x2); % s11
        c = c + a*0.39269908*gh(i1);
    end
end

stsv(1) = 2.0*c(1); % 2*pi
stsv(2) = 2.0*c(4);
stsv(3) = c(2)+c(3);
end


