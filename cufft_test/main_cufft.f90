

program cufft_r2c
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use cufft_required
    implicit none

    integer :: n
    integer :: i
    real, dimension(:), allocatable :: input
    complex, dimension(:), allocatable :: output

    ! cufft stuff
    integer(c_size_t) :: worksize(1)
    integer :: planr2c
    type(cudaLibXtDesc), pointer :: d_input, d_output
    type(cudaXtDesc), pointer    :: d_inputptr
    complex, pointer, device     :: input_dptr(:)
    integer(kind=cuda_stream_kind) :: stream

    ! example: real input, complex output
    n = 16;
    allocate(input(n))
    allocate(output(n / 2 + 1))

    print *, "Input array:"
    do i = 1, 16
        input(i) = implicit
        print*, input(i)
    end do

    ! Set cufft
    call checkCufft(cufftCreate(planr2c))
    call checkCufft(cufftMakePlan1d(planr2c, n, CUFFT_R2C, worksize), 'cufftMakePlan1d r2c error')

    ! Set device arrays
    call checkCufft(cufftXtMalloc(planr2c, d_input, CUFFT_XT_FORMAT_INPLACE), 'cufftXtMalloc error')
    call checkCufft(cufftXtMalloc(planr2c, d_output, CUFFT_XT_FORMAT_INPLACE), 'cufftXtMalloc error')

    ! copy input on gpu
    call cufft_memcpyH2D(d_input, input, CUFFT_XT_FORMAT_INPLACE, .true.)

    ! cufft
    call checkCufft(cufftExecR2C(planr2c, d_input, d_output, CUFFT_FORWARD),'forward fft failed')




    ! SUBROUTINES
    contains 

        subroutine checkCuda(istat, message)
            implicit none
            integer, intent(in)                   :: istat
            character(len=*),intent(in), optional :: message
            if (istat /= cudaSuccess) then
                write(*,"('Error code: ',I0, ': ')") istat
                write(*,*) cudaGetErrorString(istat)
                if(present(message)) write(*,*) message
                call mpi_finalize(ierr)
            endif
        end subroutine checkCuda

        subroutine checkCufft(istat, message)
            implicit none
            integer, intent(in)                   :: istat
            character(len=*),intent(in), optional :: message
            if (istat /= CUFFT_SUCCESS) then
                write(*,"('Error code: ',I0, ': ')") istat
                write(*,*) cudaGetErrorString(istat)
                if(present(message)) write(*,*) message
                call mpi_finalize(ierr)
            endif
        end subroutine checkCufft



end program cufft_r2c