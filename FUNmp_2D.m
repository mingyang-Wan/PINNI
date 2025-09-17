function [np,mp]=FUNmp_2D(ng,ne,xy,ijm,r)

%---------------------------------------------------------------------
% Function 
%  Search for MP(2,NP) and NP  located in (x-x0)^2/a^2 + (y-y0)^2/b^2=1
% Input: 
%  xy(2,ng) = nodal coordinates
%  ijm(3,ne) = element nodes
%   r=[x0,y0,a,b] = ellipse parameters 
% Output: 
%  np = number pressure boundary condition
%  mp(2,np) = nodes of mp; mp(:,1)=[1,2]', anticlockwise, consistent with original
%  triangular node order
%------------------------------------------------------------------

x0=r(1); y0=r(2); a=r(3); b=r(4);

if (a<1.0e-15)&(b<1.0e-15)
    np=0;
    mp=zeros(2,np);
    return
end

np = 0;
for i=1:ne
    kkk = 0;
    for j=1:3
        ik=ijm(j,i);
        x = xy(1,ik);
        y = xy(2,ik);
        as = ((x-x0)/a)^2 + ((y-y0)/b)^2;
        if as <= 1.0
            kkk = kkk + 1;
        end
    end
    %-----------
    if kkk == 2
        np = np + 1;
    end
end

%-------------------
mp = zeros(2,np);
g1 = zeros(1,3); k1 = 0;
g2 = zeros(1,3); k2 = 0;
ord = zeros(1,2);

n=0;
for i=1:ne
    g1 = 0;
    k1 = 0;
    g2 = 0;
    k2 = 0;
    for j=1:3
        ik=ijm(j,i);
        x = xy(1,ik);
        y = xy(2,ik);
        as = ((x-x0)/a)^2 + ((y-y0)/b)^2;
        if as <= 1.0
            k1 = k1 + 1;
            g1(k1) = j;
        else
            k2 = k2 +1;
            g2(k2) = j; 
        end
    end
    
    if k1 == 2
        n = n+1;
        if g2(1)==1
            ord=[2,3];
        else if g2(1)==2
                ord=[3,1];
            else
                ord=[1,2];
            end
        end
        mp(1,n) = ijm(ord(1),i);
        mp(2,n) = ijm(ord(2),i);
    end
end
    