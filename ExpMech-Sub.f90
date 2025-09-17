    !******************************************************************************
	subroutine mass_node(ng,ne,xy,ijm,dens, mass)   ! Jan 4, 2019
	!==============================================================================
	! Lumped nodal mass array for triangular mesh
	! Input: 
    !           ng = node number 
    !           ne = element number
	!     xy(2,ng) = node coordinates
	!    ijm(3*ne) = element node
    !         dens = solid density
	! Output:
	!        mass(ng) = nodal mass  
    ! Variable declaration:
	!       	integer,intent(in)::ng,ne,ijm(3*ne)
    !           real(8),intent(in)::xy(2*ng),dens
    !           real(8),intent(out)::mass(ng)
	!==============================================================================
		implicit none
		integer,intent(in)::ng,ne,ijm(3*ne); real(8),intent(in)::xy(2*ng),dens; real(8),intent(out)::mass(2*ng)
        real(8)::exy(6),as,vol; integer::i,j,ik,ijme(3)
        mass = 0.0 ! Initialization 
        do i=1,ne
            do j=1,3
                ik = ijm(3*i-3+j)
                ijme(j) = ik
                exy(2*j-1) = xy(2*ik-1)
                exy(2*j) = xy(2*ik)
            end do
            !------- triangular area ------
            vol = exy(1)*exy(4)-exy(2)*exy(3) + exy(3)*exy(6)-exy(4)*exy(5) + exy(5)*exy(2)-exy(6)*exy(1)
		    vol = 0.5*vol  ! triangular area 
            as = vol*dens/3.0
            do j=1,3
                ik = 2*ijme(j)
                mass(ik-1) = mass(ik-1) + as
                mass(ik) = mass(ik) + as
            end do
        end do
    end subroutine mass_node
    !******************************************************************************
	subroutine Loading_stepV1(jj, nject,nstep,loadt, dtime0,    bstep,dtime,ldt)   ! 2025-01-25 
	!==============================================================================
	! Loading step 
    !     Compared with Loadint_Step, this version deal with multiple injection point problem
	! Input: 
	!          jj = loading step
    !       nject = number of injection points
	!       nstep = total time step
    !       loadt(2,nject*nstep+nject) = loading path in different injection points, 
    !               loadt(1,:)=loading time, loadt(2,:)=loading value
    !               loadt = [injection-1| injection-2|....|injection-n]
    !       dtime0 = time interval 
	! Output:
    !        bstep = total subtime step number
    !        dtime = time interval 
    !        ldt(4,nject) = load step value at different injection point
	!                ldt(:,1)=[t0,p0,t1,p1]
    ! Variable declaration:
	!       	integer,intent(in)::jj,nject,nstep; real(8),intent(in)::loadt(2,nject*(nstep+1)),dtime0
    !           integer,intent(out)::bstep; real(8),intent(in out)::dtime,ldt(4,nject)
	!==============================================================================
		implicit none
		integer,intent(in)::jj,nject,nstep; real(8),intent(in)::loadt(2,nject*(nstep+1)),dtime0
		integer,intent(out)::bstep; real(8),intent(in out)::dtime,ldt(4,nject)
        integer::io,ini; real(8)::dt
        !------ loading time and value at t0 and t1 --------
        do io = 1,nject
            ini = (io-1)*(nstep+1)
            ldt(1,io) = loadt(1,ini + jj)     ! t0
		    ldt(2,io) = loadt(2,ini + jj)     ! load  value at t0
            ldt(3,io) = loadt(1,ini + jj+1)   ! t1
		    ldt(4,io) = loadt(2,ini + jj+1)   ! load  value at t1
        end do
        !----------- for substime step number -----------
        dt = ldt(3,1)-ldt(1,1)    ! t1-t0
        bstep = ceiling(dt/dtime0)   ! total subtime step number 
        dtime = dt/real(bstep)       ! time interval 
    end subroutine Loading_stepV1 
    !******************************************************************************
    subroutine Loading_subtime_stepV1(kk,bstep,dtime,ldt,  nject,nwd,nb,nbg,zb, nf,nfg,zf, np,npg,zp,   tim,zb1,zf1,zp1)   ! 2025-05-30
	!==============================================================================
	! Sub Load step BC
    !   Compared with the original one, this version deals with multiple injection points
	! Input: 
	!                      kk = loading step
	!                   bstep = total time step
    !                   nject = number of injection points
    !              nwd(nject) = node number at each injection point
    !            ldt(4,nject) = load step value at different injection point,ldt(:,1)=[t0,p0,t1,p1]
    !           nb,nbg,zb(nb) = restriction, variable BC number and BC value
    !           nf,nfg,zf(nf) = concentrated, variable force BC number and BC value
    !           np,npg,zp(np) = distributed,variable pressure BC number and BC value
	! Output:
    !            tim = total time
    !            dtime = time interval
    !           zb0(nb),zb1(nb) = restriction BC value at t0 and t1
    !           zp0(np),zp1(np) = distributed pressure value at t0 and t1
    !           zf0(nf),zf1(nf) = concentrated force BC value at t0 and t1
    ! Variable declaration:
	!       integer,intent(in)::kk,bstep,nject,nwd(nject),nb,nbg,nf,nfg,np,npg; real(8),intent(in)::ldt(4,nject),zb(nb),zf(nf),zp(np),dtime;
    !       real(8),intent(in out)::tim; real(8),intent(out)::zb0(nb),zb1(nb), zf0(nf),zf1(nf), zp0(np),zp1(np);
	!==============================================================================
		implicit none
		integer,intent(in)::kk,bstep,nject,nwd(nject),nb,nbg,nf,nfg,np,npg; real(8),intent(in)::ldt(4,nject),zb(nb),zf(nf),zp(np),dtime;
		real(8),intent(in out)::tim; real(8),intent(out)::zb1(nb),zf1(nf),zp1(np) !,zb0(nb),zf0(nf),zp0(np);
		integer::i,j,ini,snwd; real(8)::ldv,ldv0,cof,cof0  
        !---------------- Initialization -------------
        zb1 = zb;  ! zb0 = zb;  
        zf1 = zf;  ! zf0 = zf;
        zp1 = zp;  ! zp0 = zp; 
        !---------------------------------------------
        cof = real(kk)/real(bstep);     
        cof0 = real(kk-1)/real(bstep);
        ini = 0;
        do i=1,nject
            snwd = nwd(i)   ! number of injection nodes at i-th injection point
            tim = ldt(1,i) + cof*(ldt(3,i)-ldt(1,i));   ! time 
            ldv = ldt(2,i) + cof*(ldt(4,i)-ldt(2,i));   ! loading value
            ldv0 = ldt(2,i) + cof0*(ldt(4,i)-ldt(2,i));
            do j=1,snwd  
                ini = ini + 1
                !--------- if pressure loading ----------
		        if(npg > 0) then
                    !zp0(np-npg+ini) = ldv0;
                    zp1(np-npg+ini) = ldv;
                end if
		        !-------- if displacement loading -------
		        if(nbg > 0) then
                    !zb0(nb-nbg+ini) = ldv0;
                    zb1(nb-nbg+ini) = ldv;
                end if
		        !------------ if force loading ----------
		        if(nfg > 0) then
                    !zf0(nf-nfg+ini) = ldv0;
                    zf1(nf-nfg+ini) = ldv;
                end if
            end do
        end do
    end subroutine Loading_subtime_stepV1
    !**************************************************************************************************
	subroutine restbound(nof,ng, nb,nbg,mb,zb, u1)   ! Sep.4, 2013
	!==================================================================================================
	! Restriction boundary condition introduction
	!==================================================================================================
		implicit none
		integer,intent(in)::nof,ng,nb,nbg,mb(2,nb); real(8),intent(in)::zb(nb);real(8),intent(in out)::u1(nof*ng); 
		integer::i,ik
		    do i=1,nb
			    ik = nof*mb(1,i) - mb(2,i)
			    u1(ik) = zb(i)
		    end do
    end subroutine restbound
    !************************************************************************************
	subroutine exf_bc_2D(ng,xy, nf,mf,zf1, np,mp,zp1, exf) ! 2025-05-30
	!====================================================================================
	!   External nodal force contributed by concentrated force and pressure on Boundary 
	! Input:
    !           ng = node number 
	!     xy(2,ng) = node coordinates
    !           np,mp(2,np),zp0(np),zp1(np) = pressure BC number, infor, value at t0 and t1
    !           nf,mf(2,nf),zf0(nf),zf1(nf) = concentrated force BC number, infor, value at t0 and t1
	! Output:
 	!       exf(nof*ng) = external force
    ! Variable declaration:
	!       	integer,intent(in)::ng,nf,mf(2,nf),np,mp(2,np), npmix,mpmix(2,npmix);
    !           real(8),intent(in)::xy(2,ng),thi,dtime, zf0(nf),zf1(nf), zp0(np),zp1(np),u0(3*ng)
    !           real(8),intent(in out)::exf(3*ng)
	!====================================================================================
		implicit none
		integer,intent(in)::ng,nf,mf(2,nf),np,mp(2,np); real(8),intent(in)::xy(2,ng),zf1(nf),zp1(np)
		real(8),intent(in out)::exf(2*ng)
		integer::i,ik,n1,n2; real(8)::f1(2),dx,dy,l,n(2)
        !----------- external force/flux by force/flux boundary conditions ------------
		do i=1,nf
			ik = 2*mf(1,i)-mf(2,i)
            exf(ik) = exf(ik) + zf1(i)   ! force boundary 
		end do 
        !--------------------- external force by pressure BC --------------------------
		do i=1,np
			n1=mp(1,i); n2=mp(2,i)
			dx=xy(1,n2)-xy(1,n1); dy=xy(2,n2)-xy(2,n1)
            l = sqrt(dx*dx + dy*dy)  ! pressure distribution length 
            n(1) = dy/l; n(2)=-dx/l  ! outwards normal direction relative to the third node of triangular element 
            f1 = -0.5*l*zp1(i)*n      ! each nodal force at t1 
			exf(2*n1-1) = exf(2*n1-1) + f1(1)
			exf(2*n1) = exf(2*n1) + f1(2)
			exf(2*n2-1) = exf(2*n2-1) + f1(1)
			exf(2*n2) = exf(2*n2) + f1(2)
		end do
    end subroutine exf_bc_2D
    !************************************************************************************************************
	subroutine nodal_force(ng,ne,xy,ijm,idf, mat,u, f,stna) ! 2025-06-11
    !====================================================================================
	!   Nodal force  
    ! Input:
    !           ng,ne = node and element number 
	!           xy(2,ng) = node coordinates
    !           ijm(3*ne) = element node
    !           mat(50) = material parameters
    !           u(2*ng) = nodal disp velocity at t1
    !               idf(ne) = element state
	! In-Output:
 	!           f(2*ng) = nodal force
    !           stna(ne) = nodal force
    ! Variable declaration:
	!       	integer,intent(in)::ng,ne,ijm(3*ne),idf(ne); real(8),intent(in out)::f(2*ng),stna(ne)
    !           real(8),intent(in)::xy(2*ng),mat(50),u(2*ng)
	!====================================================================================
        implicit none
		integer,intent(in)::ng,ne,ijm(3*ne),idf(ne); real(8),intent(in out)::f(2*ng),stna(ne)
        real(8),intent(in)::xy(2*ng),mat(50),u(2*ng)
        integer::io,i,ik,ijme(3); real(8)::exy(6),eu(6),ef(6),b(3,6),vol,stnv(3),stsv(3),stn1
		f = 0.0    ! Initialization 
		do io=1,ne    ! triangular element
            if(idf(io) == 0) then
				!---------------- element infor -----------------------
				do i=1,3
					ik = ijm(3*io-3+i)
					ijme(i) = ik
					exy(2*i-1) = xy(2*ik-1)  ! node coord.
					exy(2*i)  = xy(2*ik)
					eu(2*i-1) = u(2*ik-1)   ! nodal disp
					eu(2*i) = u(2*ik)
				end do
				!------- nodal force----------
				call Bmatrix_geom_tri(exy, vol,b)  ! Geometry parameters of triangular element
				stnv = matmul(b,eu)   ! strain vector
                !-------- check whether this element is failed ------
                stn1 = stnv(1)-stnv(2)
                stn1 = 0.5*(stnv(1)+stnv(2) + sqrt(stn1*stn1 + stnv(3)*stnv(3)))
                stna(io) = stn1  ! First principal strain value
                !===================================
                !   element stress 
                !
				call stress(stnv, mat(1),mat(2),mat(3),mat(4), stsv)  ! stress vector 
				   ! stress(stnv, ym,    pr,    stnt,  stnb,   stsv)  ! stress vector
                !
                !======================================
				ef = matmul(transpose(b),stsv)*vol
  				!--------------- compile ---------------
				do i=1,3
					ik = 2*ijme(i)
					f(ik-1) = f(ik-1) + ef(2*i-1)
					f(ik) = f(ik) + ef(2*i)
                end do
            end if
		end do
    end subroutine nodal_force
    !**************************************************************************************************
	subroutine Bmatrix_geom_tri(exy, vol,b)     ! Jan 4, 2019
	!==================================================================================================
	!   B-matrix of 3-node triangular element, displacement-strain transform matrix
	!      Input:
	!           exy(6) = nodal coord
	!     Output:
	!           b(3,6) = b-matrix
	!              vol = volume of element
    ! Variable declaration:
    !         real(8),intent(in):: exy(6); 
    !         real(8),intent(out):: vol,b(3,6)
	!==================================================================================================
		implicit none
		real(8),intent(in)::exy(6); real(8),intent(out)::vol,b(3,6)
		real(8)::bt(7),as; integer::i,ik
		b = 0.0 ! Initialization
		!-------------- for geometry parameters ---------------------------------------- 
		bt(1)=exy(4)-exy(6); bt(2)=exy(6)-exy(2); bt(3)=exy(2)-exy(4); 
		bt(4)=exy(5)-exy(3); bt(5)=exy(1)-exy(5); bt(6)=exy(3)-exy(1);
		bt(7)=0.5*bt(1)*bt(5)-0.5*bt(4)*bt(2)
		vol = bt(7)
		!----------------- comput b matrix --------------------------------------------
		as=0.5/bt(7)                      ! 1/2/area of triangular element
		do i=1,3
			ik=2*i
			b(1,ik-1)=bt(i)*as;   b(1,ik)=0.0
			b(2,ik-1)=0.0; 	      b(2,ik)=bt(i+3)*as
			b(3,ik-1)=bt(i+3)*as; b(3,ik)=bt(i)*as
		end do
    end subroutine Bmatrix_geom_tri
    !**************************************************************************************************
	subroutine stress(stnv, ym,pr,stnt,stnb, stsv)   ! 2025-01-19   
	!==================================================================================================
	!  Constitutive model of  Complete AVIB model (Zhang and Gao, 2012)
    !	Compared with the original version, in this one the parameters(ym,pr,stnt, stnb) are directly transmitted rather than by Common
	!  Input:
	!     stnv(3) = strain vector
    !          ym = Young's modulus
    !          pr = Poisson ratio
    !        stnt = tensile strain strength
    !        stnb = compressive strain strength
	!  Output:
	!     stsv(3) = stress vector
    ! Variable declaration: 
    !    real(8),intent(in)::stnv(3),ym,pr,stnt,stnb; real(8),intent(out)::stsv(3)
	!==================================================================================================
		implicit none
		real(8),intent(in)::stnv(3),ym,pr,stnt,stnb; real(8),intent(out)::stsv(3)
		real(8)::gx(6),gh(6),cf(4),x1,x2,y1,y2,thi,dn,rt,c(4),a(4),stc,lama,lamb,c1,c2,q,fa,fb,d(3,3)
		integer::j1,i1; 
		data gx/-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142/
		data gh/ 0.1713244924,  0.3607615730,  0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924/
		data cf/ 0.39269908,1.178097245,1.963495085,2.74889357/
		!-------  ! pi/8         3pi/8       5pi/8     7pi/8 -------------------------------------
        d = 0.0  ! Initialize
		!===================== linear model ============================================================
  !      c1=ym/(1.0-pr*pr)
		!stsv(1) = c1*stnv(1) + c1*pr*stnv(2)
		!stsv(2) = c1*pr*stnv(1) + c1*stnv(2) 
		!stsv(3) = c1*0.5*(1.0-pr)*stnv(3)
		!return
		!===================== nonlinear model =========================================================
		lama = ym/(3.1415926*(1.0-pr)); 
		lamb = ym*(1.0-3.0*pr)/(3.1415926*(1.0-pr)*(1.0+pr))
		q = (1.0-3.0*pr)/(2.0*(1.0+pr))
		!----------------------------------------------------------------------------
		c=0.0
		do j1=1,4
			do i1=1,6 
				thi=0.39269908*gx(i1)+cf(j1)
				x1=sin(thi); x2=cos(thi)   ! micro bond direction vector x=(x1,x2)
				y1=stnv(1)*x1+0.5*stnv(3)*x2    ! y=estn*x (y1,y2)
				y2=0.5*stnv(3)*x1+stnv(2)*x2
				dn=x1*y1+x2*y2           ! bond stretch 
				rt=y1*y1+y2*y2-dn*dn     ! bond rotation^2
				stc=stnt                 ! tension      
				if(dn < 0.0) stc=-stnb   ! compression
				c1=exp(-dn/stc)
				c2=exp(-rt/(stc*stc))
				fa = lama*dn*c1*(1.0-q+q*c2)
				fb = lamb*c1*(1.0+dn/stc)*c2
				a(1) = fa*x1*x1 + fb*(y1*x1 - dn*x1*x1)   ! s11
				a(2) = fa*x1*x2 + fb*(y1*x2 - dn*x1*x2)   ! s12
				a(3) = fa*x2*x1 + fb*(y2*x1 - dn*x2*x1)   ! s21
				a(4) = fa*x2*x2 + fb*(y2*x2 - dn*x2*x2)   ! s11
				c = c + a*0.39269908*gh(i1)
			end do
		end do
		stsv(1) = 2.0*c(1)       ! 2*pi
		stsv(2) = 2.0*c(4)
		stsv(3) = c(2)+c(3)
    end subroutine stress
    !**************************************************************************************************
	subroutine stress0(stnv, ym,pr,stnt,stnb, stsv)   ! 2025-01-19   
	!==================================================================================================
	!  Constitutive model of  Complete AVIB model (Zhang and Gao, 2012)
    !	Compared with the original version, in this one the parameters(ym,pr,stnt, stnb) are directly transmitted rather than by Common
	!  Input:
	!     stnv(3) = strain vector
    !          ym = Young's modulus
    !          pr = Poisson ratio
    !        stnt = tensile strain strength
    !        stnb = compressive strain strength
	!  Output:
	!     stsv(3) = stress vector
    ! Variable declaration: 
    !    real(8),intent(in)::stnv(3),ym,pr,stnt,stnb; real(8),intent(out)::stsv(3)
	!==================================================================================================
		implicit none
		real(8),intent(in)::stnv(3),ym,pr,stnt,stnb; real(8),intent(out)::stsv(3)
		real(8)::gx(6),gh(6),cf(4),x1,x2,y1,y2,thi,dn,rt,c(4),a(4),stc,lama,lamb,c1,c2,q,fa,fb,d(3,3)
		integer::j1,i1; 
		data gx/-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142/
		data gh/ 0.1713244924,  0.3607615730,  0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924/
		data cf/ 0.39269908,1.178097245,1.963495085,2.74889357/
		!-------  ! pi/8         3pi/8       5pi/8     7pi/8 -------------------------------------
        d = 0.0  ! Initialize
		!===================== linear model ============================================================
  !      c1=ym/(1.0-pr*pr)
		!stsv(1) = c1*stnv(1) + c1*pr*stnv(2)
		!stsv(2) = c1*pr*stnv(1) + c1*stnv(2) 
		!stsv(3) = c1*0.5*(1.0-pr)*stnv(3)
		!return
		!===================== nonlinear model =========================================================
		lama = ym/(3.1415926*(1.0-pr)); 
		lamb = ym*(1.0-3.0*pr)/(3.1415926*(1.0-pr)*(1.0+pr))
		!----------------------------------------------------------------------------
		c=0.0
		do j1=1,4
			do i1=1,6 
				thi=0.39269908*gx(i1)+cf(j1)
				x1=sin(thi); x2=cos(thi)   ! micro bond direction vector x=(x1,x2)
				y1=stnv(1)*x1+0.5*stnv(3)*x2    ! y=estn*x
				y2=0.5*stnv(3)*x1+stnv(2)*x2
				dn=x1*y1+x2*y2           ! bond stretch 
				rt=y1*y1+y2*y2-dn*dn     ! bond rotation^2
                !------first derivative of bond potential-------
                fa = lama*dn
                if(dn > 0.0) fa = lama*dn*exp(-dn/stnt)
				fb = lamb*exp(-rt/stnb/stnb)
                !---------------  
				a(1) = fa*x1*x1 + fb*(y1*x1 - dn*x1*x1)   ! s11
				a(2) = fa*x1*x2 + fb*(y1*x2 - dn*x1*x2)   ! s12
				a(3) = fa*x2*x1 + fb*(y2*x1 - dn*x2*x1)   ! s21
				a(4) = fa*x2*x2 + fb*(y2*x2 - dn*x2*x2)   ! s11
				c = c + a*0.39269908*gh(i1)
			end do
		end do
		stsv(1) = 2.0*c(1)       ! 2*pi
		stsv(2) = 2.0*c(4)
		stsv(3) = c(2)+c(3)
	end subroutine	
	
    
    
    
    