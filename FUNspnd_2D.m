function [lnd,nd]=FUNspnd_2D(ng,xy,r)
%---------------------------------------------------------------------
% Function 
%  Search for specific nodes located in (x-x0)^2/a^2 + (y-y0)^2/b^2=1
% Input: 
%  xy(2,ng): nodal coordinates
%   r=[x0,y0,a,b]: ellipse parameters 
% Output: 
%  lnd = number of specific nodes
%  nd(lnd) = specific nodes
%------------------------------------------------------------------
x0=r(1); y0=r(2);
a=r(3);  b=r(4);
%----------------------------
if(a < 1.0e-15)&(b<1.0e-15)
    lnd=0;
    nd=zeros(1,lnd);
    return
end
%------------------------------

idf = zeros(1,ng);
lnd = 0;
for i=1:ng
    x = xy(1,i);
    y = xy(2,i);
    as = ((x-x0)/a)^2 + ((y-y0)/b)^2;
    if as <= 1.0 
        lnd = lnd + 1;
        idf(i) = 1;
    end
end

nd=zeros(1,lnd);
kk=0;
for i=1:ng
    if idf(i) == 1
        kk = kk + 1;
        nd(kk) = i;
    end
end
clear idf

    
