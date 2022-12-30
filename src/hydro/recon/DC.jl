#Module : 1st order reconstruction function
#Date   : 17 Dec 2022
#Author : Ka Wai HO

function DonorCellX₁!(w, wl, wr,
                      grid::AbstractGrid)
    # Get back the infromation of all dimonions
    is,ie = grid.x1.is::Int, grid.x1.ie::Int;
    js,je = grid.x2.js::Int, grid.x2.je::Int;
    ks,ke = grid.x3.ks::Int, grid.x3.ke::Int;
    Δx = 1
    copyto!( view(wl, is+1-Δx:ie+1+Δx, :, :, :), view(w, is-Δx:ie+Δx, :, :, :))
    copyto!( view(wr,   is-Δx:ie+Δx  , :, :, :), view(w, is-Δx:ie+Δx, :, :, :))
    return nothing
end

function DonorCellX₂!(w, wl, wr,
                      grid::AbstractGrid)

    # Get back the infromation of all dimonions
    is,ie = grid.x1.is::Int, grid.x1.ie::Int;
    js,je = grid.x2.js::Int, grid.x2.je::Int;
    ks,ke = grid.x3.ks::Int, grid.x3.ke::Int;
    Δx = 1
    copyto!(view(wl, :,js+1-Δx:je+1+Δx,:,:), view(w, :, js-Δx:je+Δx, :, :))
    copyto!(view(wr, :,  js-Δx:je+Δx  ,:,:), view(w, :, js-Δx:je+Δx, :, :))
    return nothing
end

function DonorCellX₃!(w, wl, wr,
                      grid::AbstractGrid)

    # Get back the infromation of all dimonions
    is,ie = grid.x1.is::Int, grid.x1.ie::Int;
    js,je = grid.x2.js::Int, grid.x2.je::Int;
    ks,ke = grid.x3.ks::Int, grid.x3.ke::Int;
    Δx = 1
    copyto!( view(wl, :, :, ks+1-Δx:ke+1+Δx, :), view(w, :, :, ks-Δx:ke+Δx, :))
    copyto!( view(wr, :, :,   ks-Δx:ke+Δx  , :), view(w, :, :, ks-Δx:ke+Δx, :))
    return nothing
end