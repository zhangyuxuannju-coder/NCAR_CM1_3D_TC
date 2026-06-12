
program SEdiag

  use netcdf
  implicit none

  INTEGER                    :: i
  REAL, PARAMETER            :: g=9.806, Dr=2000.0, Dz=500.0
  INTEGER, PARAMETER         :: Nr=121, Nz=37
  REAL, DIMENSION(Nr,Nz)     :: A, B, C, D, E, F
  REAL, DIMENSION(Nr)        :: R
  REAL, DIMENSION(Nr,0:Nz+1) :: U_se, W_se   
  REAL, DIMENSION(Nr,0:Nz+1) :: rho, psi        

    open (unit=8, file='./SE-A.ieee', form='unformatted')
         read (8) A
    close(8)
    open (unit=9, file='./SE-B.ieee', form='unformatted')
         read (9) B
    close(9)
    open (unit=10, file='./SE-C.ieee', form='unformatted')
         read (10) C
    close(10)
    open (unit=11, file='./SE-D.ieee', form='unformatted')
         read (11) D
    close(11)
    open (unit=12, file='./SE-E.ieee', form='unformatted')
         read (12) E
    close(12)
    open (unit=13, file='./SE-F.ieee', form='unformatted')
         read (13) F
    close(13)
    open (unit=14, file='./SE-rho.ieee', form='unformatted')
         read (14) rho
    close(14)

      do i=1,Nr
         R(i) = real(i-1)*Dr    
      end do


      CALL sor(Psi,A,B,C,D,E,F,Nr,Nz,Dr,Dz)
      write(6,*) ' Now calculate U_se and W_se'
      CALL psi2uw(U_se, W_se,Psi,R,rho,Dz,Dr,Nr,Nz)


    open (unit=15, file='./SE-U.ieee', form='unformatted')
         write (15) U_se
    close(15)
    open (unit=16, file='./SE-W.ieee', form='unformatted')
         write (16) W_se
    close(16)


end program SEdiag

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!--  This subroutine use Successive OverRelaxation method
!--  to solve an elliptic equation of P
!--  APxx + BPxy + CPyy + DPx + EPy = F
!  CALL sor(Psi,A,B,C,D,E,F,nr,nz,Dr,Dz)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine sor(P,A,B,C,D,E,F,Nx,Ny,Dx,Dy)

      INTEGER:: Nx, Ny, irec
      REAL:: Dx, Dy, P_xx, P_yy, P_xy, P_x, P_y, EIJ
      REAL, DIMENSION(Nx,Ny):: A, B, C, D, E, F
      REAL, DIMENSION(NX,0:Ny+1):: P

!  Local variables
      INTEGER:: max_itt, i, j, itt
      REAL:: omega, Residual, limit, max_res, Residuale
!  REAL, PARAMETER::epsilon=1.e-23
      max_itt=60000
      omega= 1.8

      limit= 1.0e-16

      write(*,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
      write(*,*)'!   Start SOR subroutine   !'
      write(*,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

      do i=1,Nx
      do j=0,Ny+1
       P(i,j)=0
      enddo
      enddo

      itt=1
      do while( itt <= max_itt )
      max_res=-1
      do i=2,Nx-1
       do j=1,Ny

         if(i.eq.Nx-1) then
            P_xx = ( P(i,j) + P(i-1,j) - 2.*P(i,j) ) / Dx**2
            P_xy = ( P(i,j+1) - P(i,j-1) - P(i-1,j+1) + P(i-1,j-1) ) / (4.*Dx*Dy)
            P_x  = ( P(i,j) - P(i-1,j) ) / (2.*Dx)
         else
            P_xx = ( P(i+1,j) + P(i-1,j) - 2.*P(i,j) ) / Dx**2
            P_xy = ( P(i+1,j+1) - P(i+1,j-1) - P(i-1,j+1) + P(i-1,j-1) ) / (4.*Dx*Dy)
            P_x  = ( P(i+1,j) - P(i-1,j) ) / (2.*Dx)
         endif
            P_yy = ( P(i,j+1) + P(i,j-1) - 2.*P(i,j) ) / Dy**2
            P_y  = ( P(i,j+1) - P(i,j-1) ) / (2.*Dy)

            Residual = A(i,j) * P_xx + B(i,j) * P_xy + C(i,j)*P_yy + D(i,j) * P_x  +  E(i,j)*P_y   - F(i,j)
            EIJ      = - 2.*A(i,j)/Dx**2 - 2.*C(i,j)/Dy**2

         P(i,j) = P(i,j) - omega*Residual/EIJ   !-- Update to new value P
         if (max_res < 0) then
             max_res=Residual
         endif
         if (Residual>abs(max_res)) then
             max_res=abs(Residual)
         endif
        enddo
      enddo

      do j=0,Ny+1
         P(Nx,j)=P(Nx-1,j)
      enddo

      if (abs(max_res)<limit) exit

      itt=itt+1

      end do

      write(*,*) "The iteration converged at itt=", itt, max_res  
      return
   end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!--This subroutine calculate u,w from toroidal function psi
!    CALL psi2uw(U_se, W_se,Psi,R,rho,Dz,Dr,Nr,Nz)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   subroutine psi2uw(U_se, W_se,Psi,R,rho,Dz,Dr,Nr,Nz)

      REAL:: R(Nr), Dz, Dr
      REAL, DIMENSION(Nr,0:Nz+1):: U_se, W_se, rho     
      REAL, DIMENSION(Nr,0:Nz+1):: Psi

      do iz=1,Nz
        do ir=2,Nr-1
          W_se(ir,iz)= (psi(ir+1,iz)-psi(ir-1,iz))/(2.*Dr*R(ir)*rho(ir,iz))
        enddo
        W_se(1,iz)=W_se(2,iz)      !±ß½çÌõ¼þ£ºdpsi/dr=0 at r=260km
        W_se(Nr,iz)=W_se(Nr-1,iz)
      enddo
      do ir=1,Nr
         W_se(ir,0)   = 0.0             
        W_se(ir,Nz+1) = 0.0
      enddo

      do ir=2,Nr
        U_se(ir,0)=-(psi(ir,1)-psi(ir,0))/(Dz*R(ir)*0.5*(rho(ir,0)+rho(ir,1)))
        U_se(ir,Nz+1)=-(psi(ir,Nz+1)-psi(ir,Nz))/(Dz*R(ir)*0.5*(rho(ir,Nz+1)+rho(ir,Nz)))
        do iz=1,Nz
          U_se(ir,iz)=-(psi(ir,iz+1)-psi(ir,iz-1))/(2.*Dz*R(ir)*rho(ir,iz))
        enddo
      enddo

      do iz=0,Nz+1
         U_se(1,iz)=0.0
      enddo
      return
   end
