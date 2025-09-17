function [area]= FUNtriArea(exy0)
% calculate triangle area
exy = zeros(1,6);
bt = zeros(1,7);
exy(1) = exy0(1,1);
exy(2) = exy0(2,1);
exy(3) = exy0(1,2);
exy(4) = exy0(2,2);
exy(5) = exy0(1,3);
exy(6) = exy0(2,3);

bt(1)=exy(4)-exy(6); bt(2)=exy(6)-exy(2); bt(3)=exy(2)-exy(4); 
bt(4)=exy(5)-exy(3); bt(5)=exy(1)-exy(5); bt(6)=exy(3)-exy(1);
bt(7)=0.5*bt(1)*bt(5)-0.5*bt(4)*bt(2);
area = bt(7);

