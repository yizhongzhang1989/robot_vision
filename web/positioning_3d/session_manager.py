"""
Session Manager for 3D Positioning Service
==========================================

Manages robot positioning sessions lifecycle.
"""

import logging
import threading
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import uuid

from models import RobotSession, View, SessionStatus, ViewStatus, CameraParams

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages positioning sessions for multiple robots.
    
    Thread-safe session storage and lifecycle management.
    """
    
    def __init__(self, session_timeout_minutes: int = 30, max_sessions: int = 100):
        """
        Initialize session manager.
        
        Args:
            session_timeout_minutes: Session timeout in minutes
            max_sessions: Maximum concurrent sessions
        """
        self.sessions: Dict[str, RobotSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.lock = threading.RLock()
        
        logger.info(f"Session manager initialized (timeout={session_timeout_minutes}min, max={max_sessions})")
    
    def create_session(
        self,
        robot_id: str,
        reference_name: str,
        num_expected_views: int
    ) -> RobotSession:
        """
        Create a new positioning session.
        
        Args:
            robot_id: Robot identifier
            reference_name: Reference image name to use
            num_expected_views: Expected number of camera views
            
        Returns:
            Created RobotSession
            
        Raises:
            ValueError: If max sessions exceeded
        """
        with self.lock:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                # Try cleanup first
                self.cleanup_sessions()
                
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError(f"Maximum concurrent sessions ({self.max_sessions}) reached")
            
            # Generate session ID
            session_id = f"{robot_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Create session
            session = RobotSession(
                session_id=session_id,
                robot_id=robot_id,
                reference_name=reference_name,
                num_expected_views=num_expected_views,
                status=SessionStatus.PENDING
            )
            
            self.sessions[session_id] = session
            
            logger.info(f"Created session {session_id} for robot '{robot_id}' ({num_expected_views} views expected)")
            
            return session
    
    def get_session(self, session_id: str) -> Optional[RobotSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            RobotSession or None if not found
        """
        with self.lock:
            return self.sessions.get(session_id)
    
    def add_view(
        self,
        session_id: str,
        view_id: str,
        image_base64: str,
        camera_params: CameraParams
    ) -> Optional[View]:
        """
        Add a view to session.
        
        Args:
            session_id: Session identifier
            view_id: View identifier
            image_base64: Image as base64 string
            camera_params: Camera parameters
            
        Returns:
            Created View or None if session not found
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return None
            
            # Create view
            view = View(
                view_id=view_id,
                session_id=session_id,
                image_base64=image_base64,
                camera_params=camera_params,
                status=ViewStatus.RECEIVED
            )
            
            # Add to session
            session.views.append(view)
            session.updated_at = datetime.now()
            
            # Update session status
            if session.status == SessionStatus.PENDING:
                session.status = SessionStatus.PROCESSING
            
            logger.info(f"Added view {view_id} to session {session_id} ({len(session.views)}/{session.num_expected_views})")
            
            return view
    
    def update_view_status(
        self,
        session_id: str,
        view_id: str,
        status: ViewStatus,
        keypoints_2d: Optional[List[Dict[str, float]]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update view status and optionally add keypoints.
        
        Args:
            session_id: Session identifier
            view_id: View identifier
            status: New view status
            keypoints_2d: Tracked keypoints (optional)
            error_message: Error message if failed (optional)
            
        Returns:
            True if updated successfully
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Find view
            view = next((v for v in session.views if v.view_id == view_id), None)
            if not view:
                return False
            
            # Update view
            view.status = status
            view.timestamp = datetime.now()
            
            if keypoints_2d:
                view.keypoints_2d = keypoints_2d
            
            if error_message:
                view.error_message = error_message
            
            # Update session timestamp
            session.updated_at = datetime.now()
            
            logger.debug(f"Updated view {view_id} status: {status.value}")
            
            return True
    
    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update session status.
        
        Args:
            session_id: Session identifier
            status: New session status
            error_message: Error message if failed (optional)
            
        Returns:
            True if updated successfully
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session.status = status
            session.updated_at = datetime.now()
            
            if error_message:
                session.error_message = error_message
            
            if status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]:
                session.completed_at = datetime.now()
            
            logger.info(f"Session {session_id} status: {status.value}")
            
            return True
    
    def list_sessions(
        self,
        robot_id: Optional[str] = None,
        status: Optional[SessionStatus] = None
    ) -> List[RobotSession]:
        """
        List sessions with optional filters.
        
        Args:
            robot_id: Filter by robot ID (optional)
            status: Filter by status (optional)
            
        Returns:
            List of matching sessions
        """
        with self.lock:
            sessions = list(self.sessions.values())
            
            if robot_id:
                sessions = [s for s in sessions if s.robot_id == robot_id]
            
            if status:
                sessions = [s for s in sessions if s.status == status]
            
            # Sort by creation time (newest first)
            sessions.sort(key=lambda s: s.created_at, reverse=True)
            
            return sessions
    
    def get_active_sessions(self) -> List[RobotSession]:
        """Get all active (not completed) sessions."""
        return self.list_sessions(
            status=None
        )
    
    def cleanup_sessions(self, force_timeout: Optional[timedelta] = None) -> int:
        """
        Clean up timed-out sessions.
        
        Args:
            force_timeout: Override default timeout (optional)
            
        Returns:
            Number of sessions cleaned up
        """
        timeout = force_timeout or self.session_timeout
        now = datetime.now()
        cleaned = 0
        
        with self.lock:
            sessions_to_remove = []
            
            for session_id, session in self.sessions.items():
                # Check timeout for non-completed sessions
                if session.status not in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
                    age = now - session.updated_at
                    
                    if age > timeout:
                        logger.warning(f"Session {session_id} timed out (age: {age})")
                        session.status = SessionStatus.TIMEOUT
                        session.completed_at = now
                        sessions_to_remove.append(session_id)
                        cleaned += 1
                
                # Remove old completed sessions (keep for 1 hour after completion)
                elif session.completed_at:
                    age = now - session.completed_at
                    if age > timedelta(hours=1):
                        logger.info(f"Removing old completed session {session_id}")
                        sessions_to_remove.append(session_id)
                        cleaned += 1
            
            # Remove sessions
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} sessions")
        
        return cleaned
    
    def get_statistics(self) -> dict:
        """Get session statistics."""
        with self.lock:
            total_sessions = len(self.sessions)
            active_sessions = len([s for s in self.sessions.values() 
                                 if s.status in [SessionStatus.PENDING, SessionStatus.PROCESSING, SessionStatus.TRIANGULATING]])
            completed_sessions = len([s for s in self.sessions.values() 
                                    if s.status == SessionStatus.COMPLETED])
            failed_sessions = len([s for s in self.sessions.values() 
                                 if s.status == SessionStatus.FAILED])
            
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'completed_sessions': completed_sessions,
                'failed_sessions': failed_sessions,
                'max_sessions': self.max_sessions
            }
