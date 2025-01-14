package datago

import (
	"fmt"
	"runtime"
	"time"
)

// Define an enum which will be used to track the state of the worker
type worker_state int

const (
	worker_idle worker_state = iota
	worker_running
	worker_done
	worker_stopping
)

// Define a custom max function
func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

// Define a stateful worker struct which will be spawned by the worker pool
type worker struct {
	state worker_state
}

// Manage a pool of workers to fetch the samples
func run_worker_pool(sampleWorker func(*worker), chanInputs *BufferedChan[SampleDataPointers], chanOutputs *BufferedChan[Sample]) {

	// Get the number of CPUs on the machine
	max_workers := runtime.NumCPU() // We suppose that this doesn´t change during the execution
	min_workers := max(int32(max_workers/8), 1)
	worker_pool_size := max_workers / 2
	const response_time = 1 * time.Second
	fmt.Println("Worker pool started. Initial size: ", worker_pool_size, " min_workers: ", min_workers, " max_workers: ", max_workers)

	// Start the workers and work on the metadata channel
	var workers []*worker

	for i := 0; i < worker_pool_size; i++ {
		new_worker := worker{state: worker_idle}
		workers = append(workers, &new_worker)
		go sampleWorker(&new_worker)
	}

	// Every second, check the state of the workers and adjust the pool size
	// based on the work backlog and idle time
	for {
		// Create new worker
		if !idle_workers(workers) && chanInputs.current_items > int32(len(workers)) && len(workers) < max_workers {
			new_worker := worker{state: worker_idle}
			workers = append(workers, &new_worker)
			go sampleWorker(&new_worker)
		}

		// Remove an existing worker
		if idle_workers(workers) && chanInputs.current_items < 2 && len(workers) > int(min_workers) {
			workers[len(workers)-1].state = worker_stopping // The goroutine will stop after the next iteration
			workers = workers[:len(workers)-1]
		}

		if done_workers(workers) {
			fmt.Println("All workers are done")
			break
		}
		time.Sleep(response_time)
	}
}

func idle_workers(workers []*worker) bool {
	// There will be some noise measuring this, it's ok, we're only interested in a big picture
	idle := 0
	for _, w := range workers {
		if w.state == worker_idle {
			idle += 1
		}
	}
	return (float64(idle) / float64(len(workers))) > 0.5
}

func done_workers(workers []*worker) bool {
	done := 0
	for _, w := range workers {
		if w.state == worker_done {
			done += 1
		}
	}
	return done == len(workers)
}
