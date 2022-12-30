"""
  MPI swaping function
  args: SwapData  : Data to be swapped
        TranData  : Data to be transferred
        src       : target node for receiving data 
        dst       : target node for sending data
        rank/comm : MPI parameters
        MPI       : MPI module
"""
function MPIDataSwaping!(SwapData, TranData,
                         src::Int, dst::Int,
                         rank, comm, MPI)
  
  # recived data from other node    
  rreq = MPI.Irecv!(SwapData, src,  src+32, comm)

  # sent data to other node
  sreq = MPI.Isend(TranData, dst, rank+32, comm);

  stats = MPI.Waitall!([rreq, sreq]);
  
  #print("$rank: Sending   $rank -> $dst = $(size(TranData))\n")
  #print("$rank: Received  $src -> $rank = $(size(SwapData))\n")

  #MPI.Barrier(comm);
  return nothing;
  
end

function MPIDataSwaping!(SwapData, TranData,
                         src::Int, dst::Int,
                         MPI)
  # Set up the MPI Parameter
  comm = MPI.COMM_WORLD;
  rank = MPI.Comm_rank(comm);
  
  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  rank, comm, MPI);

end


function MPIDataSending!(TranData,
                         dst::Int, rank::Int,
                         comm, MPI)
  # sent data to other node
  sreq = MPI.Isend(TranData, dst, rank+32, comm);

  stats = MPI.Waitall!([sreq]);
  #MPI.Barrier(comm);

  return nothing 
end

function MPIDataSending!(TranData,dst::Int,MPI)
  # Set up the MPI Parameter
  comm = MPI.COMM_WORLD;
  rank = MPI.Comm_rank(comm);

  MPIDataSending!(TranData,
                  dst, rank,
                  comm, MPI);

end