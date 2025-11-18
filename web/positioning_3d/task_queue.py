"""
Task Queue Manager for 3D Positioning Service
============================================

Manages the FFPP tracking task queue with single-threaded worker.
"""

import queue
import threading
import logging
import time
from typing import Optional, Callable
from datetime import datetime

from models import TrackingTask

logger = logging.getLogger(__name__)


class TaskQueueManager:
    """
    Manages task queue for FFPP keypoint tracking.
    
    Ensures serialized processing (1 task at a time) to avoid
    concurrent FFPP requests.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize task queue manager.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=max_size)
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.current_task: Optional[TrackingTask] = None
        self.task_handler: Optional[Callable] = None
        
        # Statistics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Task queue initialized (max_size={max_size})")
    
    def start(self, task_handler: Callable):
        """
        Start the worker thread.
        
        Args:
            task_handler: Function to call for each task
                         Should accept TrackingTask and return success boolean
        """
        if self.running:
            logger.warning("Worker thread already running")
            return
        
        self.task_handler = task_handler
        self.running = True
        
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="FFPPWorkerThread",
            daemon=True
        )
        self.worker_thread.start()
        
        logger.info("✅ Task queue worker thread started")
    
    def stop(self, timeout: float = 10.0):
        """
        Stop the worker thread gracefully.
        
        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self.running:
            return
        
        logger.info("Stopping task queue worker...")
        self.running = False
        
        # Put sentinel value to unblock queue.get()
        try:
            self.queue.put(None, block=False)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)
            
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")
            else:
                logger.info("✅ Task queue worker thread stopped")
    
    def enqueue(self, task: TrackingTask) -> bool:
        """
        Add task to queue.
        
        Args:
            task: TrackingTask to process
            
        Returns:
            True if enqueued successfully, False if queue full
        """
        try:
            self.queue.put(task, block=False)
            logger.debug(f"Task {task.task_id} enqueued (queue size: {self.get_queue_size()})")
            return True
        except queue.Full:
            logger.error(f"Queue full, cannot enqueue task {task.task_id}")
            return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def get_current_task(self) -> Optional[TrackingTask]:
        """Get currently processing task."""
        return self.current_task
    
    def get_statistics(self) -> dict:
        """Get queue statistics."""
        avg_time = (self.total_processing_time / self.tasks_processed 
                   if self.tasks_processed > 0 else 0.0)
        
        return {
            'queue_size': self.get_queue_size(),
            'tasks_processed': self.tasks_processed,
            'tasks_failed': self.tasks_failed,
            'average_processing_time': round(avg_time, 3),
            'current_task': self.current_task.to_dict() if self.current_task else None,
            'worker_running': self.running
        }
    
    def _worker_loop(self):
        """Main worker thread loop - processes tasks one at a time."""
        logger.info("Worker thread started, waiting for tasks...")
        
        while self.running:
            try:
                # Wait for next task (blocking with timeout)
                task = self.queue.get(timeout=1.0)
                
                # Check for sentinel value (stop signal)
                if task is None:
                    logger.debug("Received stop signal")
                    break
                
                # Process task
                self._process_task(task)
                
                # Mark task as done
                self.queue.task_done()
                
            except queue.Empty:
                # No tasks available, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("Worker thread exiting")
    
    def _process_task(self, task: TrackingTask):
        """
        Process a single tracking task.
        
        Args:
            task: TrackingTask to process
        """
        self.current_task = task
        start_time = time.time()
        
        logger.info(f"Processing task {task.task_id} (session: {task.session_id}, view: {task.view_id})")
        
        try:
            # Call task handler
            if self.task_handler:
                success = self.task_handler(task)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time
                
                if success:
                    self.tasks_processed += 1
                    logger.info(f"✅ Task {task.task_id} completed in {processing_time:.2f}s")
                else:
                    self.tasks_failed += 1
                    logger.warning(f"❌ Task {task.task_id} failed after {processing_time:.2f}s")
            else:
                logger.error("No task handler configured")
                self.tasks_failed += 1
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.tasks_failed += 1
            logger.error(f"Error processing task {task.task_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.current_task = None
