package datago

import (
	"runtime"
)

// Define an enum which will be used to track the state of the worker
type workerState int

const (
	workerStateIdle workerState = iota
	workerStateRunning
	workerStateDone
)

// Define a stateful worker struct which will be spawned by the worker pool
type worker struct {
	state workerState // Allows us to track the state of the worker, useful for debugging or dynamic provisioning
	done  chan bool
}

func (w *worker) stop() {
	w.state = workerStateDone
	w.done <- true
}

// Manage a pool of workers to fetch the samples
func runWorkerPool(sampleWorker func(*worker)) {
	// Shall we just use pond here ?
	// https://github.com/alitto/pond
	poolSize := runtime.NumCPU()

	// Start the workers and work on the metadata channel
	var workers []*worker

	for i := 0; i < poolSize; i++ {
		newWorker := worker{state: workerStateIdle, done: make(chan bool)}
		workers = append(workers, &newWorker)
		go sampleWorker(&newWorker)
	}

	// Wait until everyone is done
	for _, worker := range workers {
		<-worker.done
	}
}
