package datago

import (
	"runtime"
)

// Define an enum which will be used to track the state of the worker
type worker_state int

const (
	worker_idle worker_state = iota
	worker_running
	worker_done
)

// Define a stateful worker struct which will be spawned by the worker pool
type worker struct {
	state worker_state // Allows us to track the state of the worker, useful for debugging or dynamic provisioning
	done  chan bool
}

func (w *worker) stop() {
	w.state = worker_done
	w.done <- true
}

// Manage a pool of workers to fetch the samples
func runWorkerPool(sampleWorker func(*worker)) {
	// Shall we just use pond here ?
	// https://github.com/alitto/pond
	worker_pool_size := runtime.NumCPU()

	// Start the workers and work on the metadata channel
	var workers []*worker

	for i := 0; i < worker_pool_size; i++ {
		newWorker := worker{state: worker_idle, done: make(chan bool)}
		workers = append(workers, &newWorker)
		go sampleWorker(&newWorker)
	}

	// Wait until everyone is done
	for _, worker := range workers {
		<-worker.done
	}
}
