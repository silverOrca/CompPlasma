#Function to map from (x,y) to (z)
def xyToz(ix,iy,nx,ny):
    #Note: We pass nx even though we"re not using
    #it as we may want to change this mapping later
    #All the iy values at a given ix 
    #are in sequence together.
    return ix*ny + iy    