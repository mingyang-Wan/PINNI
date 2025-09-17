!****************************************************************************
!  ExpMech-Main.f90 
!  For mechanics problem with Explicit algorithm
!****************************************************************************
!****************************************************************************
! Variables  specificaiton
! nof = Number of Freedom of each node 
! nod = Number of Node of each element
! nom = Number of Dimension  
! neq = Number of Equations
!****************************************************************************
    program ExpMechMain
		implicit none
		!##############################################################################################
		!                  Define arries, matrix and variables
		!
        integer:: nom=2,nof=2,nod,neq
        integer:: ng,ne,nject,swd,nb,nbg,nf,nfg,np,npg,nstep,insp
        integer,allocatable:: ijm(:),nwd(:),wd(:),mb(:),mf(:),mp(:),idf(:)
        real(8),allocatable:: xy(:),zb(:),zf(:),zp(:),loadt(:)
        real(8)::thi,dtime0,matpara(50)
		!-------------------- computation variables --------------------------------------------- 
		real(8),allocatable::u0(:),u1(:),force(:),velo(:),exf(:),mass(:),zb1(:),zf1(:),zp1(:),ldt(:),nstn(:),stna(:)
		real(8)::tim,dtime
		integer::ii,jj,bstep,kkk,ostep
		!##############################################################################################


        !##############################################################################################
		!                open files 
		! 
        open(11,file='Pre_in.txt',status='old')
	    open(12,file='Pre_xyijmbc.txt',status='old')
        
        open(16,file='Post_idf.txt',status='old')
	    open(23,file='Post_forcen.txt',status='old')
	    open(24,file='Post_disp.txt',status='old')
	    open(25,file='Post_times.txt',status='old')
        open(29,file='Post_velocity.txt',status='old')
	    open(31,file='Post_nsts.txt',status='old')
        open(32,file='Post_straineig.txt',status='old')
		!##############################################################################################


		!##############################################################################################
		!                                    allocate discrete data
        read(11,*) ng,ne,nject,swd,nb,nbg,nf,nfg,np,npg, nstep,insp, thi,dtime0
        allocate(xy(2*ng),ijm(3*ne),idf(ne),nwd(nject),wd(swd),mb(2*nb),zb(nb),mf(2*nf),zf(nf),mp(2*np),zp(np), loadt(nject*2*(nstep+1)))
        read(12,*) xy,ijm,idf,nwd,wd,mb,zb,mf,zf,mp,zp,loadt,matpara
   		!##############################################################################################
        

		!##############################################################################################
		!
		!                       allocate mechanical arrays 
		!
        neq= nof*ng; 
        allocate(u0(neq),u1(neq),force(neq),velo(neq),exf(neq),mass(neq),ldt(4*nject),zb1(nb),zf1(nf),zp1(np),stna(ne))
        u0=0.0; u1=0.0; force=0.0; velo=0.0; exf=0.0; mass=0.0; ldt=0.0; zb1=0.0; zf1=0.0; zp1=0.0; stna=0.0
        call mass_node(ng,ne,xy,ijm,matpara(5), mass) 
        !    mass_node(ng,ne,xy,ijm,dens,       mass) 
        !##############################################################################################

        
		!##############################################################################################
		!
		!       Explicit  algorithm 
        tim = 0.0; kkk = 0; ostep = 0;
        exf = 0.0;  
		do 10 ii = 1,nstep
			print*; print*,'Mainstep ii=',ii
            call Loading_stepV1(ii, nject,nstep,loadt, dtime0,    bstep,dtime,ldt)
			do 20 jj = 1,bstep
				!--------------- BC at t1 ------------------------
                call Loading_subtime_stepV1(jj,bstep,dtime,ldt, nject,nwd, nb,nbg,zb, nf,nfg,zf, np,npg,zp,   tim,zb1,zf1,zp1)
				!--------- for nodal displacement u1 -------------------
				u1 = -0.5*dtime*force  + 1.0/dtime*mass*u0 + mass*velo + 0.5*dtime*exf  
				u1 = dtime*u1/mass
				call restbound(nof,ng, nb,nbg,mb,zb1, u1)   ! Restriction BC introduction
				!--------- for external force and nodal force at t1  ------------------
				exf = 0.0   ! External force 
                call exf_bc_2D(ng,xy, nf,mf,zf1, np,mp,zp1, exf)  ! external force induced by BC
                call nodal_force(ng,ne,xy,ijm,idf,matpara,u1, force,stna) ! nodal force
				!--------- for nodal velocity at t1 ------------------
				velo = -0.5*dtime*force + 1.0/dtime*mass*(u1-u0) + 0.5*dtime*exf   ! Nodal velocity at t1 
				velo = velo/mass
				!------------ update ----------------
				u0 = u1
				tim = tim + dtime
				!--------------- output data --------------------------------------------
                kkk = kkk + 1
                if(mod(kkk,insp) == 0) then
                    call Output_data(ng,ne, tim,u0,velo,force,stna)
                    ostep = ostep + 1
                    print*,'Outstep =',ostep 
                end if
20			continue
10		continue
        pause
    end program ExpMechMain
	!
	!                           Computation over 
	!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	include 'ExpMech-Sub.f90'      

    !******************************************************************************
	subroutine Output_data(ng,ne, tim,u,v,force,stna)   ! 2025-02-06
	!==============================================================================
	!   Output data
	!==============================================================================
		implicit none
		integer,intent(in)::ng,ne; real(8),intent(in)::tim, u(2*ng),v(2*ng),force(2*ng),stna(ne);
		!---------- output nodal displacement ------------
		write(24,*) real(u)	! disp.txt
		!------------- output nodal velocity -------------
		write(29,*) real(v) ! velocity.txt
		!----------- output nodal force ------------
		write(23,*) real(force) ! forcen.txt
		write(25,*) tim       ! times.txt
        write(32,*) real(stna) ! forcen.txt
    end subroutine Output_data


	


	

	
  
	
	




    
    
    