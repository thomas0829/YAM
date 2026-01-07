"""
LeRobot Data Saver Thread for asynchronous episode saving.
This thread continuously listens for completed episodes and saves them in LeRobot v3.0 format.
"""

import threading
import queue
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LeRobotDataSaverThread(threading.Thread):
    """
    A background thread that listens for completed episodes and saves them asynchronously
    in LeRobot v3.0 format (Parquet + MP4).
    """
    def __init__(self, lerobot_saver):
        """
        Initialize the saver thread.
        
        Args:
            lerobot_saver: Instance of LeRobotDataSaver from yam_realtime
        """
        super().__init__()
        self.lerobot_saver = lerobot_saver
        self.episode_queue = queue.Queue()  # Queue to hold episodes to save
        self.daemon = True  # Ensure the thread exits when the main program ends
        self._stop_event = threading.Event()

    def run(self):
        """
        Continuously listens for episodes to save.
        """
        logger.info("LeRobotDataSaverThread started")
        while not self._stop_event.is_set():
            try:
                # Wait for a new episode to save
                logger.debug("Waiting for episode from queue...")
                episode_buffer = self.episode_queue.get(timeout=1)
                if episode_buffer is None:  # Exit signal
                    logger.info("Received exit signal (None)")
                    break
                
                episode_index = episode_buffer.get('episode_index', 0)
                logger.info(f"Background thread: Starting to save episode {episode_index}")
                logger.info(f"Background thread: Episode has {episode_buffer.get('size', 0)} frames")
                logger.info(f"Background thread: Encoding videos (this may take 1-3 minutes)...")
                
                import time
                start_time = time.time()
                
                # Restore the episode buffer to the dataset and save it
                # This is safe because we have a deep copy
                original_buffer = self.lerobot_saver.dataset.episode_buffer
                original_started = self.lerobot_saver.episode_started
                
                try:
                    logger.info(f"Background thread: Calling dataset.save_episode() for episode {episode_index}...")
                    
                    # CRITICAL: Temporarily restore the episode buffer to dataset
                    # save_episode() without episode_data parameter will use self.episode_buffer
                    # and automatically clear it after saving
                    original_buffer = self.lerobot_saver.dataset.episode_buffer
                    self.lerobot_saver.dataset.episode_buffer = episode_buffer
                    
                    # Save episode using the buffer (this will also clear it)
                    self.lerobot_saver.dataset.save_episode()
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Background thread: Episode {episode_index} saved in {elapsed:.1f}s")
                    
                except Exception as e:
                    logger.error(f"Error saving episode {episode_index}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                finally:
                    # Mark the task as done
                    self.episode_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in LeRobotDataSaverThread: {e}")
                import traceback
                traceback.print_exc()
                # IMPORTANT: Mark task as done even on error to prevent deadlock
                try:
                    self.episode_queue.task_done()
                except:
                    pass
        
        logger.info("LeRobotDataSaverThread exiting")

    def save_episode(self, episode_buffer_copy):
        """
        Put episode buffer copy in the queue for background saving.
        
        Args:
            episode_buffer_copy: A deep copy of the episode buffer to save
        """
        if episode_buffer_copy is None:
            logger.warning("Received None episode buffer, skipping save")
            return
        self.episode_queue.put(episode_buffer_copy)
        logger.info(f"Episode buffer added to save queue (episode_index={episode_buffer_copy.get('episode_index')})")

    def stop(self):
        """Signal to stop the background thread."""
        logger.info("Stopping LeRobotDataSaverThread...")
        self._stop_event.set()
        self.episode_queue.put(None)
        
    def finalize(self):
        """Wait for all episodes to be saved and finalize the dataset."""
        logger.info("=== LeRobotDataSaverThread.finalize() called ===")
        
        # Thread has already been joined, so all episodes should be processed
        # But let's double-check the queue is empty
        if not self.episode_queue.empty():
            logger.warning(f"Queue is not empty! {self.episode_queue.qsize()} items remaining")
            logger.warning("This should not happen - data may be lost!")
        else:
            logger.info("Queue is empty, all episodes have been processed")
        
        logger.info("Calling lerobot_saver.finalize()...")
        self.lerobot_saver.finalize()
        logger.info("lerobot_saver.finalize() completed")
