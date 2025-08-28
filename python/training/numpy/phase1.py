import numpy as np

### PART 1
print("hello world!")

# 1.1
arr = np.array([1, 2, 3])
print("type : " + str(arr.dtype) + " item size: " + str(arr.itemsize))
print("*********")

# 1.2. create an array of zeros
arr = np.zeros(shape=(7, 4), dtype=np.float32)
print(f"dtype {arr.dtype} item size {arr.itemsize}")
print("*********")

# 1.3. create array of ones with shape 2x2
arr = np.ones(shape=(2, 2))
print(arr)
print(f"type & size: {arr.dtype} & {arr.itemsize}")
print("*********")

# 1.4. create array filled with constant value 7
arr = np.full((2, 2), 7)
print(arr)
print(f"type & size: {arr.dtype} & {arr.itemsize}")
print("*********")

# 1.5. create a range of values from 0 to 20 step size 5
arr = np.arange(0, 20, 5)
print(arr)
print("type: ", arr.dtype, ", size: ", arr.itemsize)
print("*********")

# 1.6
arr = np.linspace(0, 1, 11)
print(arr)
print("type: ", arr.dtype, ", size: ", arr.itemsize)
print("*********")

# 1.7
print((arr * 10.0).astype(np.int32))
print(arr.astype(np.float32) * 0.1)
print("*********")

# 2.1
arr = np.arange(12).reshape(2, 6)
print("original 2x6:\n", arr)
print("shape: ", arr.shape, "dimensions: ", arr.ndim)
arr = arr.reshape(3,4)
print("reshaped 3x4:\n", arr)
print("shape: ", arr.shape, "dimensions: ", arr.ndim)
print("*********")

# 2.2
flat = arr.ravel() # faster and less memory, is a view faster if contigious
print(flat)
print("shape: ", flat.shape)
print("*********")

# 2.3
flat = arr.flatten()    # always returns a copy, won't modify original
print(flat)
print("shape: ", flat.shape)
print("*********")

# 2.4
print("transpose:")
print(arr.T)
print("transpose (again):")
print(np.swapaxes(arr,axis1=1, axis2=0))
print("*********")

# 2.5
xarr = np.expand_dims(arr, axis=0)
print(xarr)
print(xarr.ndim)
arr = np.squeeze(xarr, axis=0)
print(arr, ": ", arr.ndim)
print("*********")

#  2.6
shp = np.shape(((2, 1), (3, 2)))
print(shp)
print("*********")

arr = np.zeros(shape=(1, 2))
print(arr[0][0])
print(arr[0][1])

arr = np.zeros(shape=(1, 2, 3))
print(arr[0][0][0])
print(arr[0][1][0])

print(arr[0][1][0])
print(arr[0][1][1])
print(arr[0][1][2])
arr = np.squeeze(arr, axis=0)
print(arr)
print("*********")

arr = np.reshape(np.arange(12), (2, 2, 3))
print(arr)
print(arr.ndim)
print(arr[0,1,1])
print("*********")

# 3.1
arr = np.arange(20).reshape(4, 5).astype(np.float32)
shp = arr.shape
print(arr)
print("*********")
# print(0, ", ", shp[1])
# v = arr[(0, shp[1]-1)]
# 3.2
rf = arr[0, :]
cl = arr[:, -1] #-1 value loops
print(rf)
print(cl)
print("*********")

# 3.3
sub = arr[1:4, 2:5] # up to, not including, so 1:n is 1 to n-1
print(sub)
print("*********")

# 3.4
slice = arr[::2, ::2]   # start:stop:step, skipping start and stop ::
print(slice)
print("*********")

# 3.5
rev = arr[::-1] # applies to only the first row, column is say, reversed because step is -1
print(rev)
print("*********")

# 3.6
view = arr[0:2, 0:2]    # this was a view, we modified it it modified the array
view[0,0] = 0.5
print(view)
print(arr)
print("*********")
cop = arr[0:2, 0:2].copy()  # faster
cop = arr.copy()[0:2, 0:2]

# 4.1
arr = np.array([10, 15, 20, 25, 30])
print(arr[[0, 2, 4]])
mask = arr > 18
print(arr[mask])
arr[mask] = 99
print(arr)

print("*********")
print("*********")
print("*********")

### TEST
# 1, 2, 3, 4
arr = np.arange(5, 25, 5)
print(arr, ", dtype=", arr.dtype)

arr = np.zeros(9, dtype=np.float32).reshape(3, 3)
print(arr)

arr = np.linspace(2,6, 6, dtype=np.float32)
print(arr)

arr = np.arange(3, dtype=np.int32).astype(np.float32)
print(arr)
print("*********")

# 5, 6, 7, 8
arr = np.ones(12).reshape(3, 4)
print(arr)
arr = np.ones(shape=[2, 3, 1])
low = np.squeeze(arr, axis=2)
print(arr.shape, " v. ", low.shape)
print("*********")

# 9 a,b,c,d
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19,20]])
e = np.ravel(arr)[12]
lr = arr[-1:]
print(arr)
print(e)
print(lr)
sub = arr[1:3, 1:4]     # row 1 <= r < 3, cols 1 <= c < 4
print(sub)
fr = arr[0:2, ::2]      # row 0 <= r < 2, cols begin:end:step 2
print(fr)
tran = np.zeros(shape=(4, 2)).T
print(tran.shape)
arr = np.full(shape=(3, 3), fill_value=7)
print(arr)
print(arr.flatten().shape)

print("*********")

# 10, 11, 12
arr = np.array([2, 7, 12, 18, 21])
mask = arr % 2 == 0
arr[mask] = -1
print(arr)
print(arr[[0, 2, 4]])
print("*********")
print("*********")
print("*********")

### MINI IMAGE TASK
rng = np.random.default_rng()
img = rng.integers(0, 256, size=(2, 3,3), dtype=np.uint8)

greens = img[:, :, 1].ravel()

print(greens)
r = img[:, :, 0].copy()
img[:, :, 0] = img[:, :, 2]
img[:, :, 2] = r

lum = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
lum[...] = 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
print(img)
print(lum)

print("goodbye!")


