import numpy as np
import os
import time

def client_random_vectors(yDim, Selector, N):
    Sel = np.zeros((yDim, 1), dtype = np.uint64)
    Sel[ Selector, 0] = 1

    vectors = []
    bytes_len = yDim * (64 / 8)

    total = np.zeros((yDim, 1), dtype = np.uint64)
    for n in range(N-1):
        x = os.urandom(bytes_len)
        R0 = np.frombuffer(x, dtype=np.uint64)
        R0 = np.resize(R0, ((yDim, 1)))
        total += R0

        vectors += [ np.getbuffer(R0) ]

    Rlast = Sel - total
    vectors += [ np.getbuffer(Rlast) ]

    return vectors, Sel

def client_process_responses(responses):
    resps = []
    for r in responses:
        r = np.frombuffer(r, dtype=np.uint64)
        # r = np.resize(r, (1, len(r)))
        resps += [ r ]


    R = resps[0]
    for r in resps[1:]:
        R += r
    return R

def server_process_request(db, request):
    request = np.frombuffer(request, dtype=np.uint64)
    request = np.resize(request, ((len(request), 1)))
    resp = db * request

    return np.getbuffer(resp)

def server_process_request_batch(db, requests):
    reqs = []
    for r in requests:
        r = np.frombuffer(r, dtype=np.uint64)
        r = np.resize(r, ((len(r), 1)))
        reqs += [ r ]

    return np.getbuffer(db * np.hstack(reqs))


def test_random_vectors():
    # Make some data
    Dim = 1024
    data = np.arange(0, Dim*Dim, 1, dtype = np.uint64)
    data = np.resize(data, (Dim, Dim))

    dataM = np.matrix(data, dtype = np.uint64)

    vectors, Sel = client_random_vectors(Dim, 10, 5)
    V1 = server_process_request(dataM, Sel)
    V1 = np.frombuffer(V1, dtype=np.uint64)

    V2 = np.zeros((Dim, 1), dtype = np.uint64)

    responses = []
    for v in vectors:
        responses += [server_process_request(dataM,v)]

    V2 = client_process_responses(responses)
    assert np.array_equal(V1, V2)

def test_random_vectors_time():
    # Make some data
    Dim = 1024

    # How many Kbytes is that?
    Mbes = Dim * Dim * 8 / (1024 * 1024)

    data = np.arange(0, Dim*Dim, 1, dtype = np.uint64)
    data = np.resize(data, (Dim, Dim))

    dataM = np.matrix(data, dtype = np.uint64)

    vectors, Sel = client_random_vectors(Dim, 123, 10)
    respose_length = len(np.getbuffer(vectors[0]))
    
    print("\n\nStats:\n------")

    print ("PIR Response length: %.2f kB" % ( float(respose_length) / ( 1024)))

    t0 = time.time()
    for _ in range(100):
        server_process_request(dataM,vectors[0])
    t1 = time.time()
    print("PIR server per lookup: %.2f ms (for a DB of %s MB)" % (1000*(t1-t0)/100.0,Mbes) )

    # Batching performance
    t0 = time.time()
    for _ in range(20):
        server_process_request_batch(dataM, vectors[:10])
    t1 = time.time()
    print("PIR server per batched lookup: %.2f ms \n\t(for a DB of %s MB, batch of 10 requests)" % (1000*(t1-t0)/(20.0 * 10.0),Mbes) )
    
