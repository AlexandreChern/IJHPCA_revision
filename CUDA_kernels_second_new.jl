# This file intends to write a unified function mfA_cuda
using CUDA

function cuda_knl_2_x_interior(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, odata)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if 2 <= i <= Ns1-1 && 2 <= j <= Nr1-1
        global_index = (i - 1) * Nr1 + j
        odata[global_index] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5*idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index-Nr1] + -0.5*idata_css[global_index]) * idata_x[global_index-Nr1]
                            + (0.5 * idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                            )
    end # needs to be further simplified

    nothing
end



function cuda_knl_2_x(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata)

    l = 2
    α = 1.0
    θ_R = 1 / 2

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if 4 <= i <= Nr1-3 && 2 <= j <= Ns1-1
        global_index = (j - 1) * Nr1 + i
        @inbounds odata[global_index] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5*idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index-Nr1] + -0.5*idata_css[global_index]) * idata_x[global_index-Nr1]
                            + (0.5 * idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                            )
    end # needs to be further simplified


    if 4 <= i <= Nr1-3 && j == 1
        global_index = i
        @inbounds odata[global_index] = (  hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + 1]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                        )
    end

    if 4 <= i <= Nr1-3 && j == Ns1
        global_index = (Ns1 - 1) * Nr1 + i
        @inbounds odata[global_index] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5 * idata_crr[global_index+1]) * idata_x[global_index+1])

                            + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + 1]))

                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + 1])))
                        )
    end

    if i == 1 && 2 <= j <= Ns1-1
        global_index = (j - 1) * Nr1 + i
        @inbounds odata[global_index] = (
                            hs* 1 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr

                            + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] - 0.5 * idata_css[global_index]) * idata_x[global_index - Nr1]
                            + (0.5*idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
                            + (-0.5*idata_css[global_index] + -0.5*idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + Nr1]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + (0.5 * idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1])
                            - 0.5 * idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1])) # Asr

                            # add modD_f1 and modD_f2 ...
                            +  (-hs * 1.0 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2]))
                            + 1.0 * 1.0 * ( 1.0 * idata_crs[global_index]) * (-0.5 * idata_x[global_index - 1 * 1 * Nr1] + 0.0 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) # Compute -L_1T⋅G_1

                            + 1.0 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[j]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 1.0 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1] - 0.5 * idata_crs[global_index + Nr1] * (1.0 * idata_x[global_index + Nr1])
                            )
                        )

        @inbounds odata[global_index + 1] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index + 1-1] + -0.5*idata_crr[global_index + 1]) * idata_x[global_index + 1-1]
                            + (0.5 * idata_crr[global_index + 1-1] + idata_crr[global_index + 1] + 0.5 * idata_crr[global_index + 1+1]) * idata_x[global_index + 1]
                            + (-0.5 * idata_crr[global_index + 1] + -0.5 * idata_crr[global_index + 1+1]) * idata_x[global_index + 1+1]) # Arr
                    
                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index + 1-Nr1] + -0.5*idata_css[global_index + 1]) * idata_x[global_index + 1-Nr1]
                            + (0.5 * idata_css[global_index + 1-Nr1] + idata_css[global_index + 1] + 0.5 * idata_css[global_index + 1+Nr1]) * idata_x[global_index + 1]
                            + (-0.5 * idata_css[global_index + 1] + -0.5 * idata_css[global_index + 1+Nr1]) * idata_x[global_index + 1+Nr1]) # Ass

                            + (0.5 * idata_crs[global_index + 1 - 1] * (-0.5 * idata_x[global_index + 1 - Nr1 - 1] + 0.5 * idata_x[global_index + 1 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1 + 1] * (-0.5 * idata_x[global_index + 1 - Nr1 + 1] + 0.5 * idata_x[global_index + 1 + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index + 1 - Nr1] * (-0.5 * idata_x[global_index + 1 - 1 - Nr1] + 0.5 * idata_x[global_index + 1 + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index + 1 + Nr1] * (-0.5 * idata_x[global_index + 1 - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + 1 + Nr1]))) # Asr

                            -hs * 1.0 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                        )

        @inbounds odata[global_index + 2] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index + 2-1] + -0.5*idata_crr[global_index + 2]) * idata_x[global_index + 2-1]
                            + (0.5 * idata_crr[global_index + 2-1] + idata_crr[global_index + 2] + 0.5 * idata_crr[global_index + 2+1]) * idata_x[global_index + 2]
                            + (-0.5 * idata_crr[global_index + 2] + -0.5 * idata_crr[global_index + 2+1]) * idata_x[global_index + 2+1]) # Arr
                            
                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index + 2-Nr1] + -0.5*idata_css[global_index + 2]) * idata_x[global_index + 2-Nr1]
                            + (0.5 * idata_css[global_index + 2-Nr1] + idata_css[global_index + 2] + 0.5 * idata_css[global_index + 2+Nr1]) * idata_x[global_index + 2]
                            + (-0.5 * idata_css[global_index + 2] + -0.5 * idata_css[global_index + 2+Nr1]) * idata_x[global_index + 2+Nr1]) # Ass
                            
                            + (0.5 * idata_crs[global_index + 2 - 1] * (-0.5 * idata_x[global_index + 2 - Nr1 - 1] + 0.5 * idata_x[global_index + 2 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 2 + 1] * (-0.5 * idata_x[global_index + 2 - Nr1 + 1] + 0.5 * idata_x[global_index + 2 + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index + 2 - Nr1] * (-0.5 * idata_x[global_index + 2 - 1 - Nr1] + 0.5 * idata_x[global_index + 2 + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index + 2 + Nr1] * (-0.5 * idata_x[global_index + 2 - 1 + Nr1] + 0.5 * idata_x[global_index + 2 + 1 + Nr1]))) # Asr

                            -hs * 1.0 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                        )
    end

    if i == Nr1 && 2 <= j <= Ns1-1
        global_index = j * Nr1

        @inbounds odata[global_index] = ( hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                            + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index - Nr1] + -0.5 * idata_css[global_index]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index-1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1]) 
                            + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + Nr1])) # Ars

                            + (0.5 * idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index - Nr1])
                            - 0.5 * idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + Nr1])) # Asr

                            + (-hs * 1.0 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2]))
                            - 1.0 * 1.0 * ( 1.0 * idata_crs[global_index]) * (-0.5 * idata_x[global_index - 1 * 1 * Nr1] + 0.0 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) # Compute -L_1T⋅G_1

                            + 1.0 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[j]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 1.0 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            - 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1] + 0.5 * idata_crs[global_index + Nr1] * (1.0 * idata_x[global_index + Nr1])
                            )
                        )

        @inbounds odata[global_index - 1] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index - 1-1] + -0.5*idata_crr[global_index - 1]) * idata_x[global_index - 1-1]
                            + (0.5 * idata_crr[global_index - 1-1] + idata_crr[global_index - 1] + 0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1]
                            + (-0.5 * idata_crr[global_index - 1] + -0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1+1]) # Arr
                            
                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index - 1-Nr1] + -0.5*idata_css[global_index - 1]) * idata_x[global_index - 1-Nr1]
                            + (0.5 * idata_css[global_index - 1-Nr1] + idata_css[global_index - 1] + 0.5 * idata_css[global_index - 1+Nr1]) * idata_x[global_index - 1]
                            + (-0.5 * idata_css[global_index - 1] + -0.5 * idata_css[global_index - 1+Nr1]) * idata_x[global_index - 1+Nr1]) # Ass
                            
                            + (0.5 * idata_crs[global_index - 1 - 1] * (-0.5 * idata_x[global_index - 1 - Nr1 - 1] + 0.5 * idata_x[global_index - 1 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index - 1 + 1] * (-0.5 * idata_x[global_index - 1 - Nr1 + 1] + 0.5 * idata_x[global_index - 1 + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index - 1 - Nr1] * (-0.5 * idata_x[global_index - 1 - 1 - Nr1] + 0.5 * idata_x[global_index - 1 + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index - 1 + Nr1] * (-0.5 * idata_x[global_index - 1 - 1 + Nr1] + 0.5 * idata_x[global_index - 1 + 1 + Nr1]))) # Asr

                            -hs * 1.0 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                        )

        @inbounds odata[global_index - 2] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index - 2-1] + -0.5*idata_crr[global_index - 2]) * idata_x[global_index - 2-1]
                            + (0.5 * idata_crr[global_index - 2-1] + idata_crr[global_index - 2] + 0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2]
                            + (-0.5 * idata_crr[global_index - 2] + -0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2+1]) # Arr
                            
                            + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index - 2-Nr1] + -0.5*idata_css[global_index - 2]) * idata_x[global_index - 2-Nr1]
                            + (0.5 * idata_css[global_index - 2-Nr1] + idata_css[global_index - 2] + 0.5 * idata_css[global_index - 2+Nr1]) * idata_x[global_index - 2]
                            + (-0.5 * idata_css[global_index - 2] + -0.5 * idata_css[global_index - 2+Nr1]) * idata_x[global_index - 2+Nr1]) # Ass
                            
                            + (0.5 * idata_crs[global_index - 2 - 1] * (-0.5 * idata_x[global_index - 2 - Nr1 - 1] + 0.5 * idata_x[global_index - 2 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index - 2 + 1] * (-0.5 * idata_x[global_index - 2 - Nr1 + 1] + 0.5 * idata_x[global_index - 2 + Nr1 + 1])) # Ars
                            
                            + (0.5 * (idata_crs[global_index - 2 - Nr1] * (-0.5 * idata_x[global_index - 2 - 1 - Nr1] + 0.5 * idata_x[global_index - 2 + 1 - Nr1]))
                            + -0.5 * (idata_crs[global_index - 2 + Nr1] * (-0.5 * idata_x[global_index - 2 - 1 + Nr1] + 0.5 * idata_x[global_index - 2 + 1 + Nr1]))) # Asr

                            -hs * 1.0 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                        )
    end

    if i == 1 && j == 1
        global_index = (j - 1) * Nr1 + i# Arr

        @inbounds odata[global_index] = (hs* 0.5 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr
                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + -hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2])) # Compute -L_1T⋅G_1
                            + 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index] + 1 * idata_x[global_index + Nr1])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + -0.5 * idata_crs[global_index + Nr1] * 1.0 * idata_x[global_index + Nr1]
                            + -1 * idata_crs[global_index] * 0.5 * idata_x[global_index]

                            + hr * 0.5 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                            + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index ] + 0.5 * idata_x[global_index + 1]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                        )

        @inbounds odata[global_index + 1] = ( hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index + 1 - 1] + -0.5 * idata_crr[global_index + 1]) * idata_x[global_index + 1 - 1]
                            + (0.5 * idata_crr[global_index + 1 - 1] + idata_crr[global_index + 1] + 0.5 * idata_crr[global_index + 1 + 1]) * idata_x[global_index + 1]
                            + (-0.5 * idata_crr[global_index + 1] + -0.5 * idata_crr[global_index + 1 + 1]) * idata_x[global_index + 1 + 1]) # Arr
                            
                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index + 1] + 0.5 * idata_css[global_index + 1 + Nr1]) * idata_x[global_index + 1]
                            + (-0.5 * idata_css[global_index + 1] + -0.5 * idata_css[global_index + 1 + Nr1]) * idata_x[global_index + 1 + Nr1]) # Ass
                            
                            + (0.5 * idata_crs[global_index + 1 - 1] * (-0.5 * idata_x[global_index + 1 - 1] + 0.5 * idata_x[global_index + 1 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1 + 1] * (-0.5 * idata_x[global_index + 1 + 1] + 0.5 * idata_x[global_index + 1 + Nr1 + 1])) # Ars
                            
                            + (-0.5 * (idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1 - 1] + 0.5 * idata_x[global_index + 1 + 1]))
                            + -0.5 * (idata_crs[global_index + 1 + Nr1] * (-0.5 * idata_x[global_index + 1 - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + 1 + Nr1]))) # Asr

                            -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                        )
        
        @inbounds odata[global_index + 2] = (  hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index + 2 - 1] + -0.5 * idata_crr[global_index + 2]) * idata_x[global_index + 2 - 1]
                            + (0.5 * idata_crr[global_index + 2-1] + idata_crr[global_index + 2] + 0.5 * idata_crr[global_index + 2 + 1]) * idata_x[global_index + 2]
                            + (-0.5 * idata_crr[global_index + 2] + -0.5 * idata_crr[global_index + 2 + 1]) * idata_x[global_index + 2 + 1]) # Arr
                            
                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index + 2] + 0.5 * idata_css[global_index + 2 + Nr1]) * idata_x[global_index + 2]
                            + (-0.5 * idata_css[global_index + 2] + -0.5 * idata_css[global_index + 2 + Nr1]) * idata_x[global_index + 2 + Nr1]) # Ass
                            
                            + (0.5 * idata_crs[global_index + 2 - 1] * (-0.5 * idata_x[global_index + 2 - 1] + 0.5 * idata_x[global_index + 2 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 2 + 1] * (-0.5 * idata_x[global_index + 2 + 1] + 0.5 * idata_x[global_index + 2 + Nr1 + 1])) # Ars
                            
                            + (-0.5 * (idata_crs[global_index + 2] * (-0.5 * idata_x[global_index + 2 - 1] + 0.5 * idata_x[global_index + 2 + 1]))
                            + -0.5 * (idata_crs[global_index + 2 + Nr1] * (-0.5 * idata_x[global_index + 2 - 1 + Nr1] + 0.5 * idata_x[global_index + 2 + 1 + Nr1]))) # Asr

                            -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                        )
    end


    if i == Nr1  && j == 1
        global_index = (j - 1) * Nr1 + i# Arr
        @inbounds odata[global_index] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + Nr1 - 1]) 
                            + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1])) # Ars

                            +  (-hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2])) # Compute -L_1T⋅G_1
                            - 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index] + 1 * idata_x[global_index + Nr1])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[j]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index + Nr1] * 1.0 * idata_x[global_index + Nr1]
                            + 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]

                            + hr * 0.5 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass
    
                            + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + Nr1]))) # Asr
                            )
                        )
        
        @inbounds odata[global_index - 1] = (  hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index - 1-1] + -0.5 * idata_crr[global_index - 1]) * idata_x[global_index - 1-1]
                            + (0.5 * idata_crr[global_index - 1-1] + idata_crr[global_index - 1] + 0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1]
                            + (-0.5 * idata_crr[global_index - 1] + -0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1+1]) # Arr

                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index - 1] + 0.5 * idata_css[global_index - 1 + Nr1]) * idata_x[global_index - 1]
                            + (-0.5 * idata_css[global_index - 1] + -0.5 * idata_css[global_index - 1 + Nr1]) * idata_x[global_index - 1 + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 1 - 1] * (-0.5 * idata_x[global_index - 1 - 1] + 0.5 * idata_x[global_index - 1 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index - 1 + 1] * (-0.5 * idata_x[global_index - 1 + 1] + 0.5 * idata_x[global_index - 1 + Nr1 + 1])) # Ars

                            + (-0.5 * (idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1 - 1] + 0.5 * idata_x[global_index - 1 + 1]))
                            + -0.5 * (idata_crs[global_index - 1 + Nr1] * (-0.5 * idata_x[global_index - 1 - 1 + Nr1] + 0.5 * idata_x[global_index - 1 + 1 + Nr1]))) # Asr

                            -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                        )

        @inbounds odata[global_index - 2] = (  hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index - 2-1] + -0.5 * idata_crr[global_index - 2]) * idata_x[global_index - 2-1]
                            + (0.5 * idata_crr[global_index - 2-1] + idata_crr[global_index - 2] + 0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2]
                            + (-0.5 * idata_crr[global_index - 2] + -0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2+1]) # Arr

                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index - 2] + 0.5 * idata_css[global_index - 2 + Nr1]) * idata_x[global_index - 2]
                            + (-0.5 * idata_css[global_index - 2] + -0.5 * idata_css[global_index - 2 + Nr1]) * idata_x[global_index - 2 + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 2 - 1] * (-0.5 * idata_x[global_index - 2 - 1] + 0.5 * idata_x[global_index - 2 + Nr1 - 1])
                            - 0.5 * idata_crs[global_index - 2 + 1] * (-0.5 * idata_x[global_index - 2 + 1] + 0.5 * idata_x[global_index - 2 + Nr1 + 1])) # Ars

                            + (-0.5 * (idata_crs[global_index - 2] * (-0.5 * idata_x[global_index - 2 - 1] + 0.5 * idata_x[global_index - 2 + 1]))
                            + -0.5 * (idata_crs[global_index - 2 + Nr1] * (-0.5 * idata_x[global_index - 2 - 1 + Nr1] + 0.5 * idata_x[global_index - 2 + 1 + Nr1]))) # Asr

                            -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                        )

        
    end

    if i == 1  && j == Ns1
        global_index = (j - 1) * Nr1 + i

        @inbounds odata[global_index] = (hs* 0.5 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr

                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + 1])) # Ars

                            + -hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2])) # Compute -L_1T⋅G_1
                            + 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index - Nr1] + 1 * idata_x[global_index])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[j]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1]
                            + 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]

                            + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 
                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + 1])))
                        )

        @inbounds odata[global_index + 1] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index + 1-1] + -0.5 * idata_crr[global_index + 1]) * idata_x[global_index + 1 - 1]
                        + (0.5 * idata_crr[global_index + 1 - 1] + idata_crr[global_index + 1] + 0.5 * idata_crr[global_index + 1 + 1]) * idata_x[global_index + 1]
                        + (-0.5 * idata_crr[global_index + 1] - 0.5 * idata_crr[global_index + 1 + 1]) * idata_x[global_index + 1 + 1])
                        
                        + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index + 1 - Nr1] + -0.5 * idata_css[global_index + 1]) * idata_x[global_index + 1 - Nr1]
                        + (0.5 * idata_css[global_index + 1 - Nr1] + 0.5 * idata_css[global_index + 1]) * idata_x[global_index  + 1]) 
                        
                        + (0.5 * idata_crs[global_index + 1 - 1] * (-0.5 * idata_x[global_index + 1 - Nr1 - 1] + 0.5 * idata_x[global_index + 1 - 1])
                        - 0.5 * idata_crs[global_index + 1 + 1] * (-0.5 * idata_x[global_index + 1 - Nr1 + 1] + 0.5 * idata_x[global_index + 1 + 1]))
                        
                        + (0.5 * (idata_crs[global_index + 1 - Nr1] * (-0.5 * idata_x[global_index + 1 - 1 - Nr1] + 0.5 * idata_x[global_index + 1 + 1 - Nr1]))
                        + 0.5 * (idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1 - 1] + 0.5 * idata_x[global_index + 1 + 1])))

                        -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                        )

        @inbounds odata[global_index + 2] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index + 2-1] + -0.5 * idata_crr[global_index + 2]) * idata_x[global_index + 2-1]
                        + (0.5 * idata_crr[global_index + 2-1] + idata_crr[global_index + 2] + 0.5 * idata_crr[global_index + 2+1]) * idata_x[global_index + 2]
                        + (-0.5 * idata_crr[global_index + 2] - 0.5 * idata_crr[global_index + 2+1]) * idata_x[global_index + 2+1])
                        
                        + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index + 2-Nr1] + -0.5 * idata_css[global_index + 2 ]) * idata_x[global_index + 2 - Nr1]
                        + (0.5 * idata_css[global_index + 2 - Nr1] + 0.5 * idata_css[global_index + 2]) * idata_x[global_index + 2]) 
                        
                        + (0.5 * idata_crs[global_index + 2 - 1] * (-0.5 * idata_x[global_index + 2 - Nr1 - 1] + 0.5 * idata_x[global_index + 2 - 1])
                        - 0.5 * idata_crs[global_index + 2 + 1] * (-0.5 * idata_x[global_index + 2 - Nr1 + 1] + 0.5 * idata_x[global_index + 2 + 1]))
                        
                        + (0.5 * (idata_crs[global_index + 2 - Nr1] * (-0.5 * idata_x[global_index + 2 - 1 - Nr1] + 0.5 * idata_x[global_index + 2 + 1 - Nr1]))
                        + 0.5 * (idata_crs[global_index + 2] * (-0.5 * idata_x[global_index + 2 - 1] + 0.5 * idata_x[global_index + 2 + 1])))

                        -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                        )

    end


    if i == Nr1  && j == Ns1
        global_index = (j - 1) * Nr1 + i

        @inbounds odata[global_index] = (
                        hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                        + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                        + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1-1] + 0.5 * idata_x[global_index-1]) 
                        + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index])) # Ars

                        + (-hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2])) # Compute -L_1T⋅G_1
                        - 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index - Nr1] + 1 * idata_x[global_index])

                        + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                        - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                        - 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1]
                        - 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]
                        )
                        
                        + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                        + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 
                        + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index - Nr1]))
                        + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index])))

                    )

        
        @inbounds odata[global_index - 1] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index - 1-1] + -0.5 * idata_crr[global_index - 1]) * idata_x[global_index - 1-1]
                        + (0.5 * idata_crr[global_index - 1-1] + idata_crr[global_index - 1] + 0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1]
                        + (-0.5 * idata_crr[global_index - 1] - 0.5 * idata_crr[global_index - 1+1]) * idata_x[global_index - 1+1])
                        
                        + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index - 1-Nr1] + -0.5 * idata_css[global_index - 1 ]) * idata_x[global_index - 1 - Nr1]
                        + (0.5 * idata_css[global_index - 1 - Nr1] + 0.5 * idata_css[global_index - 1]) * idata_x[global_index - 1]) 
                        
                        + (0.5 * idata_crs[global_index - 1 - 1] * (-0.5 * idata_x[global_index - 1 - Nr1 - 1] + 0.5 * idata_x[global_index - 1 - 1])
                        - 0.5 * idata_crs[global_index - 1 + 1] * (-0.5 * idata_x[global_index - 1 - Nr1 + 1] + 0.5 * idata_x[global_index - 1 + 1]))
                        
                        + (0.5 * (idata_crs[global_index - 1 - Nr1] * (-0.5 * idata_x[global_index - 1 - 1 - Nr1] + 0.5 * idata_x[global_index - 1 + 1 - Nr1]))
                        + 0.5 * (idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1 - 1] + 0.5 * idata_x[global_index - 1 + 1])))

                        -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
                    )

        @inbounds odata[global_index - 2] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index - 2-1] + -0.5 * idata_crr[global_index - 2]) * idata_x[global_index - 2-1]
                        + (0.5 * idata_crr[global_index - 2-1] + idata_crr[global_index - 2] + 0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2]
                        + (-0.5 * idata_crr[global_index - 2] - 0.5 * idata_crr[global_index - 2+1]) * idata_x[global_index - 2+1])
                        
                        + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index - 2-Nr1] + -0.5 * idata_css[global_index - 2 ]) * idata_x[global_index - 2 - Nr1]
                        + (0.5 * idata_css[global_index - 2 - Nr1] + 0.5 * idata_css[global_index - 2]) * idata_x[global_index - 2]) 
                        
                        + (0.5 * idata_crs[global_index - 2 - 1] * (-0.5 * idata_x[global_index - 2 - Nr1 - 1] + 0.5 * idata_x[global_index - 2 - 1])
                        - 0.5 * idata_crs[global_index - 2 + 1] * (-0.5 * idata_x[global_index - 2 - Nr1 + 1] + 0.5 * idata_x[global_index - 2 + 1]))
                        
                        + (0.5 * (idata_crs[global_index - 2 - Nr1] * (-0.5 * idata_x[global_index - 2 - 1 - Nr1] + 0.5 * idata_x[global_index - 2 + 1 - Nr1]))
                        + 0.5 * (idata_crs[global_index - 2] * (-0.5 * idata_x[global_index - 2 - 1] + 0.5 * idata_x[global_index - 2 + 1])))

                        -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
                    )
    end


    nothing
end



function cuda_knl_2_x_f1(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata)
    l = 2
    α = 1.0
    θ_R = 1 / 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if 2 <= i <= Ns1-1
        global_index = (i-1)*Nr1 + 1 # global index
        odata[global_index] = (
                            hs* 1 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr

                            + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] - 0.5 * idata_css[global_index]) * idata_x[global_index - Nr1]
                            + (0.5*idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
                            + (-0.5*idata_css[global_index] + -0.5*idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + Nr1]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + (0.5 * idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1])
                            - 0.5 * idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1])) # Asr

                            ## add modD_f1 and modD_f2 ...
                            +  (-hs * 1.0 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2]))
                            + 1.0 * 1.0 * ( 1.0 * idata_crs[global_index]) * (-0.5 * idata_x[global_index - 1 * 1 * Nr1] + 0.0 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) # Compute -L_1T⋅G_1

                            + 1.0 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 1.0 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1] - 0.5 * idata_crs[global_index + Nr1] * (1.0 * idata_x[global_index + Nr1])
                            )
                        )
        odata[global_index + 1] = -hs * 1.0 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index + 2] = -hs * 1.0 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    if i == 1 
        global_index = (i - 1) * Nr1 + 1 # Arr
        odata[global_index] = (hs* 0.5 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr
                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + -hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2])) # Compute -L_1T⋅G_1
                            + 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index] + 1 * idata_x[global_index + Nr1])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + -0.5 * idata_crs[global_index + Nr1] * 1.0 * idata_x[global_index + Nr1]
                            + -1 * idata_crs[global_index] * 0.5 * idata_x[global_index]
                        )
        odata[global_index + 1] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index + 2] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    if i == Ns1
        global_index = (i - 1) * Nr1 + 1 # Arr
        odata[global_index] = (hs* 0.5 * (1/hr) * ((0.5 * idata_crr[global_index] + 0.5*idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5*idata_crr[global_index+1])*idata_x[global_index+1]) # Arr

                            + (-0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index]) 
                            + -0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + 1])) # Ars

                            + -hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + 2])) # Compute -L_1T⋅G_1
                            + 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index - Nr1] + 1 * idata_x[global_index])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ1[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1]
                            + 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]
                        )
        odata[global_index + 1] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index + 2] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    nothing
end


function cuda_knl_2_x_f2(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata)
    l = 2
    α = 1.0
    θ_R = 1 / 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= Ns1-1
        global_index = i * Nr1
        odata[global_index] = ( hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                            + hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index - Nr1] + -0.5 * idata_css[global_index]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index-1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1]) 
                            + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + Nr1])) # Ars

                            + (0.5 * idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index - Nr1])
                            - 0.5 * idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + Nr1])) # Asr

                            + (-hs * 1.0 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2]))
                            - 1.0 * 1.0 * ( 1.0 * idata_crs[global_index]) * (-0.5 * idata_x[global_index - 1 * 1 * Nr1] + 0.0 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1]) # Compute -L_1T⋅G_1

                            + 1.0 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 1.0 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            - 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1] + 0.5 * idata_crs[global_index + Nr1] * (1.0 * idata_x[global_index + Nr1])
                            )
                        )
        odata[global_index - 1] = -hs * 1.0 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index - 2] = -hs * 1.0 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    if i == 1
        global_index = (i-1)*Nr1 + Nr1 # global index
        odata[global_index] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + Nr1 - 1]) 
                            + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + Nr1])) # Ars

                            +  (-hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2])) # Compute -L_1T⋅G_1
                            - 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index] + 1 * idata_x[global_index + Nr1])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            + 0.5 * idata_crs[global_index + Nr1] * 1.0 * idata_x[global_index + Nr1]
                            + 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]
                            )
                        )
        odata[global_index - 1] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index - 2] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    if i == Ns1
        global_index = i * Nr1
        odata[global_index] = (
                            hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + 0.5 * idata_crr[global_index]) * idata_x[global_index]) # Arr

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1-1] + 0.5 * idata_x[global_index-1]) 
                            + 0.5 * idata_crs[global_index] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index])) # Ars

                            + (-hs * 0.5 * (1/hr) * (idata_crr[global_index] * (1.5 * idata_x[global_index] -2.0 * idata_x[global_index - 1] + 0.5 * idata_x[global_index - 2])) # Compute -L_1T⋅G_1
                            - 0.5 * idata_crs[global_index] * (-1 * idata_x[global_index - Nr1] + 1 * idata_x[global_index])

                            + 0.5 * idata_crr[global_index] * ((2/θ_R) + (1/α) * (idata_crr[global_index]) / idata_ψ2[i]) * idata_x[global_index] # Compute M̃ += L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]

                            - hs * 0.5 * (idata_crr[global_index] * ((1/hr) * 1.5 * idata_x[global_index])) # Compute M̃ -= G[f]' * L[f] 
                            - 0.5 * idata_crs[global_index - Nr1] * 1.0 * idata_x[global_index - Nr1]
                            - 1 * idata_crs[global_index] * 0.5 * idata_x[global_index]
                            )
                        )
        odata[global_index - 1] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * -2.0 * idata_x[global_index])
        odata[global_index - 2] = -hs * 0.5 * (idata_crr[global_index] * (1/hr) * 0.5 * idata_x[global_index])
    end

    nothing               
end



function cuda_knl_2_x_f3(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata)
    l = 2
    α = 1.0
    θ_R = 1 / 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= Nr1-1
        global_index = i
        odata[global_index] = (  hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

                            + hr * 1.0 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                            + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars

                            + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + 1]))
                            + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                        )
    end

    if i == 1
        global_index = i
        odata[global_index] = ( hr * 0.5 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                        + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                        + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index ] + 0.5 * idata_x[global_index + 1]))
                        + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
                        )
    end

    if i == Nr1
        global_index = i
        odata[global_index] = ( hr * 0.5 * (1/hs) * ((0.5 * idata_css[global_index] + 0.5 * idata_css[global_index + Nr1]) * idata_x[global_index]
                        + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index + Nr1]) * idata_x[global_index + Nr1]) # Ass

                        + (-0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index]))
                        + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + Nr1]))) # Asr
                    )
    end

    nothing               
end

function cuda_knl_2_x_f4(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata)
    l = 2
    α = 1.0
    θ_R = 1 / 2
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if 2 <= i <= Nr1-1
        global_index = (Ns1 - 1) * Nr1 + i
        odata[global_index] = (hs * 0.5 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5 * idata_crr[global_index]) * idata_x[global_index-1]
                            + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
                            + (-0.5 * idata_crr[global_index] - 0.5 * idata_crr[global_index+1]) * idata_x[global_index+1])

                            + hr * 1.0 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 

                            + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index - 1])
                            - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + 1]))

                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index + 1])))
                        )
    end

    if i == 1
        global_index = (Ns1 - 1) * Nr1 + i
        odata[global_index] = (hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 
                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
                            + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index] + 0.5 * idata_x[global_index + 1])))
                        )
    end

    if i == Nr1
        global_index = (Ns1 - 1) * Nr1 + i
        odata[global_index] = ( hr * 0.5 * (1/hs) * ((-0.5 * idata_css[global_index-Nr1] + -0.5 * idata_css[global_index ]) * idata_x[global_index - Nr1]
                            + (0.5 * idata_css[global_index - Nr1] + 0.5 * idata_css[global_index]) * idata_x[global_index]) 
                            + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index - Nr1]))
                            + 0.5 * (idata_crs[global_index] * (-0.5 * idata_x[global_index - 1] + 0.5 * idata_x[global_index])))
                        )
    end
    nothing
end


function cuda_knl_2_H_inverse(hr, hs, idata_x, Nr1, Ns1, odata)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # hx = 1/(Nr1 - 1)
    # hy = 1/(Ns1 - 1)

    if 2 <= i <= Ns1-1 && 2 <= j <= Nr1-1
        global_index = (i - 1) * Nr1 + j
        # odata[global_index] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5*idata_crr[global_index]) * idata_x[global_index-1]
        #                     + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
        #                     + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

        #                     + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index-Nr1] + -0.5*idata_css[global_index]) * idata_x[global_index-Nr1]
        #                     + (0.5 * idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
        #                     + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

        #                     + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
        #                     - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars
                            
        #                     + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
        #                     + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
        #                     )
        odata[global_index] = idata_x[global_index] / (hr * hs)
    end 

    if (i == 1 || i == Ns1) && (j == 1 || j == Nr1)
        global_index = (i - 1) * Nr1 + j
        odata[global_index] = idata_x[global_index] * 4 / (hr * hs)
    end 

    if ( ((i == 1 || i == Ns1) && (2 <= j <= Nr1 - 1)) || ((2 <= i <= Ns1 - 1) && (j == 1 || j == Nr1)) ) 
        global_index = (i - 1) * Nr1 + j
        odata[global_index] = idata_x[global_index] * 2 / (hr * hs)
    end

    nothing
end


function cuda_knl_2_H(hr, hs, idata_x, Nr1, Ns1, odata)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # hx = 1/(Nr1 - 1)
    # hy = 1/(Ns1 - 1)

    if 2 <= i <= Ns1-1 && 2 <= j <= Nr1-1
        global_index = (i - 1) * Nr1 + j
        # odata[global_index] = (hs * 1.0 * (1/hr) * ((-0.5 * idata_crr[global_index-1] + -0.5*idata_crr[global_index]) * idata_x[global_index-1]
        #                     + (0.5 * idata_crr[global_index-1] + idata_crr[global_index] + 0.5 * idata_crr[global_index+1]) * idata_x[global_index]
        #                     + (-0.5 * idata_crr[global_index] + -0.5 * idata_crr[global_index+1]) * idata_x[global_index+1]) # Arr

        #                     + hs * 1.0 * (1/hr) * ((-0.5 * idata_css[global_index-Nr1] + -0.5*idata_css[global_index]) * idata_x[global_index-Nr1]
        #                     + (0.5 * idata_css[global_index-Nr1] + idata_css[global_index] + 0.5 * idata_css[global_index+Nr1]) * idata_x[global_index]
        #                     + (-0.5 * idata_css[global_index] + -0.5 * idata_css[global_index+Nr1]) * idata_x[global_index+Nr1]) # Ass

        #                     + (0.5 * idata_crs[global_index - 1] * (-0.5 * idata_x[global_index - Nr1 - 1] + 0.5 * idata_x[global_index + Nr1 - 1])
        #                     - 0.5 * idata_crs[global_index + 1] * (-0.5 * idata_x[global_index - Nr1 + 1] + 0.5 * idata_x[global_index + Nr1 + 1])) # Ars
                            
        #                     + (0.5 * (idata_crs[global_index - Nr1] * (-0.5 * idata_x[global_index - 1 - Nr1] + 0.5 * idata_x[global_index + 1 - Nr1]))
        #                     + -0.5 * (idata_crs[global_index + Nr1] * (-0.5 * idata_x[global_index - 1 + Nr1] + 0.5 * idata_x[global_index + 1 + Nr1]))) # Asr
        #                     )
        odata[global_index] = idata_x[global_index] * (hr * hs)
    end 

    if (i == 1 || i == Ns1) && (j == 1 || j == Nr1)
        global_index = (i - 1) * Nr1 + j
        odata[global_index] = idata_x[global_index] * (hr * hs) / 4
    end 

    if ( ((i == 1 || i == Ns1) && (2 <= j <= Nr1 - 1)) || ((2 <= i <= Ns1 - 1) && (j == 1 || j == Nr1)) ) 
        global_index = (i - 1) * Nr1 + j
        odata[global_index] = idata_x[global_index] * (hr * hs) / 2
    end

    nothing
end



function prolongation_2D_kernel(idata,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy


    if 1 <= i <= Nx-1 && 1 <= j <= Ny-1
        odata[2*i-1,2*j-1] = idata[i,j]
        odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
        odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
        odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
    end 

    if 1 <= j <= Ny-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end

    if 1 <= i <= Nx-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end

    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_prolongation_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim prolongation_2D_kernel(idata,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing
end

function restriction_2D_kernel(idata,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy

    # idata = CuArray(zeros(Nx+2,Ny+2))
    # idata[2:end-1,2:end-1] .= idata

    size_odata = (div(Nx+1,2),div(Ny+1,2))

    if 2 <= i <= size_odata[1] - 1 && 2 <= j <= size_odata[2] - 1
        odata[i,j] = (4*idata[2*i-1,2*j-1]
        + 2 * (idata[2*i,2*j-1] + idata[2*i-2,2*j-1] + idata[2*i-1,2*j] + idata[2*i-1,2*j-2])
        + (idata[2*i-2,2*j-2] + idata[2*i,2*j] + idata[2*i-2,2*j]) + idata[2*i,2*j-2]
         ) / 16
    end

    if i == 1 && j == 1
        odata[i,j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
    end

    if i == size_odata[1] && j == 1
        odata[i,j] = (idata[2*i-1,2*j-1] + idata[2*i-2,2*j-1] + idata[2*i-1,2*j] + idata[2*i-2,2*j]) / 4
    end

    if i == 1 && j == size_odata[2]
        odata[i,j] = (idata[i,2*j-1] + idata[i+1,2*j-1] + idata[i,2*j-2] + idata[i+1,2*j-2]) / 4
    end

    if i == size_odata[1] && j == size_odata[2]
        odata[i,j] = (idata[2*i-1,2*j-1] + idata[2*i-2,2*j-1] + idata[2*i-1,2*j-2] + idata[2*i-2,2*j-2]) / 4
    end

    if 2 <= i <= size_odata[1] - 1 && j == 1
        odata[i,j] = (2 * idata[2*i-1,2*j-1] + idata[2*i-2,2*j-1] + idata[2*i,2*j-1]
        + 2 * idata[2*i-1,2*j] + idata[2*i-2,2*j] + idata[2*i,2*j]
        ) / 8
    end

    if 2 <= i <= size_odata[1] - 1 && j == size_odata[2]
        odata[i,j] = (2 * idata[2*i-1,2*j-1] + idata[2*i-2,2*j-1] + idata[2*i,2*j-1]
        + 2 * idata[2*i-1,2*j-2] + idata[2*i-2,2*j-2] + idata[2*i,2*j-2]
        ) / 8
    end

    if i == 1 && 2 <= j <= size_odata[2] - 1
        odata[i,j] = (2 * idata[2*i-1,2*j-1] + idata[2*i-1,2*j-2] + idata[2*i-1,2*j]
        + 2 * idata[2*i,2*j-1] + idata[2*i,2*j-2] + idata[2*i,2*j]
        ) / 8
    end

    if i == size_odata[1] && 2 <= j <= size_odata[2] - 1
        odata[i,j] = (2 * idata[2*i-1,2*j-1] + idata[2*i-1,2*j-2] + idata[2*i-1,2*j]
        + 2 * idata[2*i-2,2*j-1] + idata[2*i-2,2*j-2] + idata[2*i-2,2*j]
        ) / 8
    end

    # This restriction operator is not consistent with the matrix-explicit ones 
    # FIxing this one could help with performance
   
    return nothing
end

function matrix_free_restriction_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    # idata = CuArray(zeros(Nx+2,Ny+2))
    # copyto!(view(idata,2:Nx+1,2:Ny+1),idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim restriction_2D_kernel(idata,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing

end