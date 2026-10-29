program fdtd_coarray_optimized
    use iso_fortran_env
    implicit none

    ! Parameters
    integer, parameter :: Ni = 512, Nj = 512, Nk = 512
    integer, parameter :: num_iterations = 25
    real(8), parameter :: C = 3e10, PI = 3.14159265358
    real(8), parameter :: dx = C, dy = C, dz = C, dt = 0.2
    real(8), parameter :: coef_B_dx = C * dt / (2 * dx), coef_B_dy = C * dt / (2 * dy), coef_B_dz = C * dt / (2 * dz)
    real(8), parameter :: coef_E_dx = C * dt / dx, coef_E_dy = C * dt / dy, coef_E_dz = C * dt / dz
    real(8), parameter :: coef_J = 4*PI*dt

    ! Field arrays (optimized layout for vectorization)
    real(8), allocatable :: Ex(:,:,:)[:], Ey(:,:,:)[:], Ez(:,:,:)[:]
    real(8), allocatable :: Bx(:,:,:)[:], By(:,:,:)[:], Bz(:,:,:)[:]
    real(8), allocatable :: Jx(:,:,:)[:], Jy(:,:,:)[:], Jz(:,:,:)[:]
    real(8), allocatable :: next_Ex(:,:)[:], next_Ey(:,:)[:], pred_Bx(:,:)[:], pred_By(:,:)[:]

    ! Local variables
    integer :: this_img, total_imgs, next_img, pred_img
    integer :: k_start, k_end, k_local
    integer :: idx, t
    integer(int64) :: start_time, end_time, rate
    integer :: start_i, start_j, start_k, max_i, max_j, max_k
    real(8) :: elapsed_time
    real(8) :: Tx, Ty, Tz, TT, bnd, current_time
    
    ! Coarray initialization
    this_img = this_image()
    total_imgs = num_images()
    next_img = merge(this_image() + 1, 1, this_image() < num_images())
    pred_img = merge(this_image() - 1, num_images(), this_image() > 1)

    TT = 8.0
    Tx = 4 * C
    Ty = 4 * C
    Tz = 4 * C
    current_time = min(int(TT / dt), num_iterations)

    bnd = Ni / 2.0 * dx

    start_i = floor((-Tx/4.0 + bnd) / dx) + 1
    start_j = floor((-Ty/4.0 + bnd) / dy) + 1
    start_k = floor((-Tz/4.0 + bnd) / dz) + 1

    max_i = floor((Tx/4.0 + bnd) / dx) + 1
    max_j = floor((Ty/4.0 + bnd) / dy) + 1
    max_k = floor((Tz/4.0 + bnd) / dz) + 1
    
    ! Domain decomposition
    call init_decomposition()
    
    ! Allocate arrays with optimized layout
    allocate(Ex(Ni, Nj, k_local)[*])
    allocate(Ey(Ni, Nj, k_local)[*])
    allocate(Ez(Ni, Nj, k_local)[*])
    allocate(Bx(Ni, Nj, k_local)[*])
    allocate(By(Ni, Nj, k_local)[*])
    allocate(Bz(Ni, Nj, k_local)[*])
    allocate(Jx(Ni, Nj, k_local)[*])
    allocate(Jy(Ni, Nj, k_local)[*])
    allocate(Jz(Ni, Nj, k_local)[*])

    allocate(next_Ex(Ni, Nj)[*])
    allocate(next_Ey(Ni, Nj)[*])
    allocate(pred_Bx(Ni, Nj)[*])
    allocate(pred_By(Ni, Nj)[*])
    
    ! Initialize fields
    call init_fields()
    
    ! Timing and output
    if (this_img == 1) then
        print '(A, I0)', 'Running on ', total_imgs, ' images'
        call system_clock(count_rate=rate)
        call system_clock(count=start_time)
    end if
    
    ! Main FDTD loop
    do t = 1, current_time
        if (this_img == 1) then
            call init_currents(t)
        end if
        sync all

        call update_B_field()
        sync all

        pred_Bx(:,:) = Bx(:, :, k_local)[pred_img]
        pred_By(:,:) = By(:, :, k_local)[pred_img]
        sync all
        
        call update_E_field()
        sync all
        
        next_Ex(:,:) = Ex(:, :, 1)[next_img]
        next_Ey(:,:) = Ey(:, :, 1)[next_img]
        sync all

        call update_B_field()
        sync all

    end do
    
    Jx = 0.0
    Jy = 0.0
    Jz = 0.0
    sync all

    do t = current_time + 1, num_iterations
        call update_B_field()
        sync all

        pred_Bx(:,:) = Bx(:, :, k_local)[pred_img]
        pred_By(:,:) = By(:, :, k_local)[pred_img]
        sync all

        call update_E_field()
        sync all

        next_Ex(:,:) = Ex(:, :, 1)[next_img]
        next_Ey(:,:) = Ey(:, :, 1)[next_img]
        sync all

        call update_B_field()
        sync all

    end do

    sync all
    ! Final timing and cleanup
    if (this_img == 1) then
        call system_clock(count=end_time)
        elapsed_time = real(end_time - start_time)/real(rate)
        print '(A, F0.2, A)', 'Total execution time: ', elapsed_time, ' seconds'
    end if

    sync all
    if ((this_image() == 1)) then !.and. (Nk <= 16)) then
        call print_full_E_slice()
    end if
    
    deallocate(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz)
    deallocate(next_Ex, next_Ey, pred_Bx, pred_By)

contains
    !===============================================================
    subroutine init_decomposition
        integer :: remainder, offset
        
        remainder = mod(Nk, total_imgs)
        k_local = Nk / total_imgs
        
        if (this_img <= remainder) then
            k_local = k_local + 1
            offset = (this_img - 1)*k_local
        else
            offset = remainder*(k_local + 1) + (this_img - remainder - 1)*k_local
        end if
        
        k_start = offset + 1
        k_end = offset + k_local
    end subroutine init_decomposition

    !===============================================================
    subroutine init_fields
        Ex = 0.0
        Ey = 0.0
        Ez = 0.0
        next_Ex = 0.0
        next_Ey = 0.0
        pred_Bx = 0.0
        pred_By = 0.0
        Bx = 0.0
        By = 0.0
        Bz = 0.0
        Jx = 0.0
        Jy = 0.0
        Jz = 0.0
    end subroutine init_fields

    !===============================================================
    subroutine init_currents(this_t)
        integer, intent(in) :: this_t
        integer :: i, j, k
        integer :: this_cur_img
        real(8) :: value

        do k = start_k, max_k
            do j = start_j, max_j
                do i = start_i, max_i
                    this_cur_img = ceiling(real(k) / real(k_local))
                    value = (sin(2.0 * PI * this_t * dt / TT)) &
                          * (cos(2.0 * PI * (i-1) * dx / Tx)**2) &
                          * (cos(2.0 * PI * (j-1) * dy / Ty)**2) &
                          * (cos(2.0 * PI * (k-1) * dz / Tz)**2)
                    Jx(i,j,k - (this_cur_img - 1) * k_local)[this_cur_img] = value
                    Jy(i,j,k - (this_cur_img - 1) * k_local)[this_cur_img] = value
                    Jz(i,j,k - (this_cur_img - 1) * k_local)[this_cur_img] = value
                end do
            end do
        end do
    end subroutine init_currents

    !===============================================================
    subroutine update_B_field
        integer :: ip, jp, kp
        integer :: i, j, k

        do k = 1, k_local - 1
            do j = 1, Nj
                jp = merge(j+1, 1, j < Nj)
                do i = 1, Ni
                    ip = merge(i+1, 1, i < Ni)
	                    
                    Bx(i,j,k) = Bx(i,j,k) + coef_B_dz * (Ey(i,j,k+1) - Ey(i,j,k)) - &
                                            coef_B_dy * (Ez(i,jp,k) - Ez(i,j,k))
                    
                    By(i,j,k) = By(i,j,k) + coef_B_dx * (Ez(ip,j,k) - Ez(i,j,k)) - &
                                            coef_B_dz * (Ex(i,j,k+1) - Ex(i,j,k))
                    
                    Bz(i,j,k) = Bz(i,j,k) + coef_B_dy * (Ex(i,jp,k) - Ex(i,j,k)) - &
                                            coef_B_dx * (Ey(ip,j,k) - Ey(i,j,k))
                end do
            end do
        end do
        do j = 1, Nj
            jp = merge(j+1, 1, j < Nj)
            do i = 1, Ni
                ip = merge(i+1, 1, i < Ni)

                Bx(i,j,k_local) = Bx(i,j,k_local) + coef_B_dz * (next_Ey(i,j) - Ey(i,j,k_local)) - &
                                                    coef_B_dy * (Ez(i,jp,k_local) - Ez(i,j,k_local))

                By(i,j,k_local) = By(i,j,k_local) + coef_B_dx * (Ez(ip,j,k_local) - Ez(i,j,k)) - &
                                                    coef_B_dz * (next_Ex(i,j) - Ex(i,j,k_local))

                Bz(i,j,k_local) = Bz(i,j,k_local) + coef_B_dy * (Ex(i,jp,k_local) - Ex(i,j,k_local)) - &
                                                    coef_B_dx * (Ey(ip,j,k_local) - Ey(i,j,k_local))
            end do
        end do

    end subroutine update_B_field

    !===============================================================
    subroutine update_E_field
        integer :: im, jm, km
        integer :: i, j, k

        do j = 1, Nj
            jm = merge(j-1, Nj, j > 1)
            do i = 1, Ni
                im = merge(i-1, Ni, i > 1)

                Ex(i,j,1) = Ex(i,j,1) - coef_J * Jx(i,j,1) + &
                            coef_E_dy * (Bz(i,j,1) - Bz(i,jm,1)) - &
                            coef_E_dz * (By(i,j,1) - pred_By(i,j))

                Ey(i,j,1) = Ey(i,j,1) - coef_J * Jy(i,j,1) + &
                            coef_E_dz * (Bx(i,j,1) - pred_Bx(i,j)) - &
                            coef_E_dx * (Bz(i,j,1) - Bz(im,j,1))

                Ez(i,j,1) = Ez(i,j,1) - coef_J * Jz(i,j,1) + &
                            coef_E_dx * (By(i,j,1) - By(im,j,1)) - &
                            coef_E_dy * (Bx(i,j,1) - Bx(i,jm,1))
            end do
        end do
        do k = 2, k_local
            do j = 1, Nj
                jm = merge(j-1, Nj, j > 1)
                do i = 1, Ni
                    im = merge(i-1, Ni, i > 1)
                    
                    Ex(i,j,k) = Ex(i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dy * (Bz(i,j,k) - Bz(i,jm,k)) - &
                                coef_E_dz * (By(i,j,k) - By(i,j,k-1))

                    Ey(i,j,k) = Ey(i,j,k) - coef_J * Jy(i,j,k) + &
                                coef_E_dz * (Bx(i,j,k) - Bx(i,j,k-1)) - &
                                coef_E_dx * (Bz(i,j,k) - Bz(im,j,k))
                    
                    Ez(i,j,k) = Ez(i,j,k) - coef_J * Jz(i,j,k) + &
                                coef_E_dx * (By(i,j,k) - By(im,j,k)) - &
                                coef_E_dy * (Bx(i,j,k) - Bx(i,jm,k))
                end do
            end do
        end do
    end subroutine update_E_field

    !===============================================================
subroutine print_full_E_slice()
    integer :: img, ik, ij
    do ij = Nj/2-4, Nj/2+5
        do ik = Ni/2-4, Ni/2+5
            img = ceiling(real(Nk/2+1) / real(k_local))
            write(*, '(F12.5)', advance='no')  Ex(ik,ij,Nk/2+1 - (img - 1) * k_local)[img]
        end do
        print *
    end do

end subroutine print_full_E_slice
end program fdtd_coarray_optimized
