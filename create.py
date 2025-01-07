
import ngtpy
import sys
n = len(sys.argv)
print(sys.argv)
anngIndex = sys.argv[1]
n = int(sys.argv[2])
idx = ngtpy.Index(path=anngIndex)
print("create anng: ", anngIndex, n)
idx.batch_insert_path(anngIndex, n, num_threads=1, debug=False)
idx.save()
idx.close()