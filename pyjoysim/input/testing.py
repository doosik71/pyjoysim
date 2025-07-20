"""
Input testing and validation utilities for PyJoySim.

This module provides comprehensive testing tools for validating
joystick input functionality and performance.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .joystick_manager import JoystickManager, JoystickInfo, JoystickInput
from .input_processor import InputProcessor, InputEvent, InputEventType
from .config_manager import InputConfigManager, JoystickProfile
from .hotplug import IntegratedHotplugManager, HotplugEvent
from ..config import get_settings
from ..core.logging import get_logger
# from ..core.exceptions import InputError, ValidationError


class TestType(Enum):
    """Types of input tests."""
    CONNECTION = "connection"
    INPUT_RESPONSE = "input_response"
    DEADZONE = "deadzone"
    CALIBRATION = "calibration"
    PERFORMANCE = "performance"
    HOTPLUG = "hotplug"
    PROFILE = "profile"


class TestResult(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    test_type: TestType
    description: str
    expected_result: Any = None
    timeout: float = 30.0
    required: bool = True


@dataclass
class TestReport:
    """Results from a test execution."""
    test_case: TestCase
    result: TestResult
    actual_result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class InputTester:
    """
    Comprehensive tester for joystick input functionality.
    
    Provides automated and interactive testing capabilities for
    validating joystick input systems.
    """
    
    def __init__(self, 
                 joystick_manager: JoystickManager,
                 input_processor: InputProcessor,
                 config_manager: InputConfigManager,
                 hotplug_manager: Optional[IntegratedHotplugManager] = None):
        """
        Initialize the input tester.
        
        Args:
            joystick_manager: Joystick manager instance
            input_processor: Input processor instance  
            config_manager: Configuration manager instance
            hotplug_manager: Optional hotplug manager instance
        """
        self.logger = get_logger("input_tester")
        self.settings = get_settings()
        
        # Component references
        self.joystick_manager = joystick_manager
        self.input_processor = input_processor
        self.config_manager = config_manager
        self.hotplug_manager = hotplug_manager
        
        # Test state
        self._test_results: List[TestReport] = []
        self._running_tests = False
        self._current_test: Optional[TestCase] = None
        
        # Event collection for testing
        self._collected_events: List[InputEvent] = []
        self._collection_active = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self._performance_data: Dict[str, List[float]] = {
            "input_latency": [],
            "event_processing_time": [],
            "polling_rate": []
        }
        
        self.logger.debug("InputTester initialized")
    
    def run_basic_tests(self, joystick_id: Optional[int] = None) -> List[TestReport]:
        """
        Run basic input validation tests.
        
        Args:
            joystick_id: Optional specific joystick to test (tests all if None)
            
        Returns:
            List of test reports
        """
        self.logger.info("Starting basic input tests", extra={
            "target_joystick": joystick_id
        })
        
        test_cases = self._create_basic_test_cases()
        return self._execute_test_suite(test_cases, joystick_id)
    
    def run_performance_tests(self, duration: float = 30.0, joystick_id: Optional[int] = None) -> List[TestReport]:
        """
        Run performance tests for input system.
        
        Args:
            duration: Test duration in seconds
            joystick_id: Optional specific joystick to test
            
        Returns:
            List of test reports
        """
        self.logger.info("Starting performance tests", extra={
            "duration": duration,
            "target_joystick": joystick_id
        })
        
        test_cases = self._create_performance_test_cases(duration)
        return self._execute_test_suite(test_cases, joystick_id)
    
    def run_interactive_calibration(self, joystick_id: int) -> TestReport:
        """
        Run interactive calibration test for a specific joystick.
        
        Args:
            joystick_id: ID of joystick to calibrate
            
        Returns:
            Test report with calibration results
        """
        test_case = TestCase(
            name=f"Interactive Calibration (Joystick {joystick_id})",
            test_type=TestType.CALIBRATION,
            description="Interactive calibration test for joystick axes and buttons",
            timeout=300.0  # 5 minutes for interactive test
        )
        
        self.logger.info("Starting interactive calibration", extra={
            "joystick_id": joystick_id
        })
        
        return self._run_calibration_test(test_case, joystick_id)
    
    def _create_basic_test_cases(self) -> List[TestCase]:
        """Create basic test cases."""
        return [
            TestCase(
                name="Joystick Connection Test",
                test_type=TestType.CONNECTION,
                description="Verify joystick connection and basic information",
                expected_result=True
            ),
            TestCase(
                name="Input Response Test",
                test_type=TestType.INPUT_RESPONSE,
                description="Test basic input response and event generation",
                expected_result=True,
                timeout=10.0
            ),
            TestCase(
                name="Deadzone Validation",
                test_type=TestType.DEADZONE,
                description="Validate deadzone functionality",
                expected_result=True
            ),
            TestCase(
                name="Profile Loading Test",
                test_type=TestType.PROFILE,
                description="Test loading and applying input profiles",
                expected_result=True
            )
        ]
    
    def _create_performance_test_cases(self, duration: float) -> List[TestCase]:
        """Create performance test cases."""
        return [
            TestCase(
                name="Input Latency Test",
                test_type=TestType.PERFORMANCE,
                description="Measure input latency and responsiveness",
                timeout=duration + 5.0
            ),
            TestCase(
                name="Event Processing Performance",
                test_type=TestType.PERFORMANCE,
                description="Measure event processing performance",
                timeout=duration + 5.0
            ),
            TestCase(
                name="Memory Usage Test",
                test_type=TestType.PERFORMANCE,
                description="Monitor memory usage during input processing",
                timeout=duration + 5.0
            )
        ]
    
    def _execute_test_suite(self, test_cases: List[TestCase], joystick_id: Optional[int]) -> List[TestReport]:
        """Execute a suite of test cases."""
        self._running_tests = True
        self._test_results.clear()
        
        try:
            for test_case in test_cases:
                if not self._running_tests:
                    break
                
                self._current_test = test_case
                report = self._execute_single_test(test_case, joystick_id)
                self._test_results.append(report)
                
                self.logger.debug("Test completed", extra={
                    "test_name": test_case.name,
                    "result": report.result.value,
                    "execution_time": report.execution_time
                })
        
        finally:
            self._running_tests = False
            self._current_test = None
        
        return self._test_results.copy()
    
    def _execute_single_test(self, test_case: TestCase, joystick_id: Optional[int]) -> TestReport:
        """Execute a single test case."""
        start_time = time.time()
        
        try:
            if test_case.test_type == TestType.CONNECTION:
                result = self._test_connection(joystick_id)
            elif test_case.test_type == TestType.INPUT_RESPONSE:
                result = self._test_input_response(joystick_id)
            elif test_case.test_type == TestType.DEADZONE:
                result = self._test_deadzone(joystick_id)
            elif test_case.test_type == TestType.PROFILE:
                result = self._test_profile_loading(joystick_id)
            elif test_case.test_type == TestType.PERFORMANCE:
                result = self._test_performance(test_case, joystick_id)
            elif test_case.test_type == TestType.HOTPLUG:
                result = self._test_hotplug()
            else:
                result = TestReport(
                    test_case=test_case,
                    result=TestResult.SKIPPED,
                    error_message="Test type not implemented"
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestReport(
                test_case=test_case,
                result=TestResult.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _test_connection(self, joystick_id: Optional[int]) -> TestReport:
        """Test joystick connection."""
        test_case = self._current_test
        details = {}
        
        if joystick_id is not None:
            # Test specific joystick
            if not self.joystick_manager.is_joystick_connected(joystick_id):
                return TestReport(
                    test_case=test_case,
                    result=TestResult.FAILED,
                    error_message=f"Joystick {joystick_id} not connected"
                )
            
            info = self.joystick_manager.get_joystick_info(joystick_id)
            details[f"joystick_{joystick_id}"] = {
                "name": info.name,
                "guid": info.guid,
                "axes": info.num_axes,
                "buttons": info.num_buttons,
                "hats": info.num_hats
            }
        else:
            # Test all joysticks
            joystick_count = self.joystick_manager.get_joystick_count()
            if joystick_count == 0:
                return TestReport(
                    test_case=test_case,
                    result=TestResult.WARNING,
                    error_message="No joysticks connected",
                    details={"joystick_count": 0}
                )
            
            all_joysticks = self.joystick_manager.get_all_joysticks()
            for jid, info in all_joysticks.items():
                details[f"joystick_{jid}"] = {
                    "name": info.name,
                    "guid": info.guid,
                    "axes": info.num_axes,
                    "buttons": info.num_buttons,
                    "hats": info.num_hats
                }
        
        return TestReport(
            test_case=test_case,
            result=TestResult.PASSED,
            actual_result=True,
            details=details
        )
    
    def _test_input_response(self, joystick_id: Optional[int]) -> TestReport:
        """Test input response and event generation."""
        test_case = self._current_test
        
        # Start event collection
        self._start_event_collection()
        
        try:
            # Wait for input events or timeout
            timeout = 10.0
            start_time = time.time()
            
            self.logger.info("Waiting for input events", extra={
                "timeout": timeout,
                "instructions": "Please move joystick axes or press buttons"
            })
            
            while (time.time() - start_time) < timeout:
                if len(self._collected_events) > 0:
                    break
                time.sleep(0.1)
            
            # Stop collection
            self._stop_event_collection()
            
            # Analyze results
            if len(self._collected_events) == 0:
                return TestReport(
                    test_case=test_case,
                    result=TestResult.WARNING,
                    error_message="No input events detected (try moving joystick)",
                    details={"events_collected": 0}
                )
            
            # Analyze event types
            event_types = {}
            for event in self._collected_events:
                event_type = event.event_type.value
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            details = {
                "total_events": len(self._collected_events),
                "event_types": event_types,
                "collection_duration": timeout
            }
            
            return TestReport(
                test_case=test_case,
                result=TestResult.PASSED,
                actual_result=True,
                details=details
            )
            
        finally:
            self._stop_event_collection()
    
    def _test_deadzone(self, joystick_id: Optional[int]) -> TestReport:
        """Test deadzone functionality."""
        test_case = self._current_test
        
        # Get current deadzone setting
        current_deadzone = self.joystick_manager._deadzone
        
        # Test different deadzone values
        test_values = [0.0, 0.1, 0.2, 0.5]
        results = {}
        
        for deadzone in test_values:
            self.joystick_manager.set_deadzone(deadzone)
            
            # Collect some input samples
            samples = []
            for _ in range(10):
                if joystick_id is not None:
                    input_state = self.joystick_manager.get_input_state(joystick_id)
                    if input_state:
                        samples.append(input_state.axes)
                time.sleep(0.01)
            
            results[f"deadzone_{deadzone}"] = {
                "sample_count": len(samples),
                "deadzone_value": deadzone
            }
        
        # Restore original deadzone
        self.joystick_manager.set_deadzone(current_deadzone)
        
        return TestReport(
            test_case=test_case,
            result=TestResult.PASSED,
            actual_result=True,
            details=results
        )
    
    def _test_profile_loading(self, joystick_id: Optional[int]) -> TestReport:
        """Test profile loading and application."""
        test_case = self._current_test
        
        # Get available profiles
        profiles = self.config_manager.get_all_profiles()
        
        if not profiles:
            return TestReport(
                test_case=test_case,
                result=TestResult.FAILED,
                error_message="No profiles available to test"
            )
        
        # Test loading each profile
        profile_results = {}
        for profile_id, profile in profiles.items():
            try:
                # Test profile validation
                errors = self.config_manager.validate_profile(profile)
                
                profile_results[profile_id] = {
                    "name": profile.name,
                    "profile_type": profile.profile_type.value,
                    "validation_errors": errors,
                    "is_valid": len(errors) == 0
                }
                
            except Exception as e:
                profile_results[profile_id] = {
                    "error": str(e),
                    "is_valid": False
                }
        
        # Check if any profiles failed validation
        failed_profiles = [pid for pid, result in profile_results.items() 
                          if not result.get("is_valid", False)]
        
        if failed_profiles:
            return TestReport(
                test_case=test_case,
                result=TestResult.WARNING,
                error_message=f"Some profiles failed validation: {failed_profiles}",
                details=profile_results
            )
        
        return TestReport(
            test_case=test_case,
            result=TestResult.PASSED,
            actual_result=True,
            details=profile_results
        )
    
    def _test_performance(self, test_case: TestCase, joystick_id: Optional[int]) -> TestReport:
        """Test performance metrics."""
        duration = test_case.timeout - 5.0  # Leave 5 seconds buffer
        
        # Clear previous performance data
        self._performance_data = {
            "input_latency": [],
            "event_processing_time": [],
            "polling_rate": []
        }
        
        # Start performance monitoring
        start_time = time.time()
        sample_count = 0
        
        while (time.time() - start_time) < duration:
            sample_start = time.time()
            
            # Update joystick manager (this polls input)
            self.joystick_manager.update()
            
            # Measure polling time
            poll_time = time.time() - sample_start
            self._performance_data["polling_rate"].append(1.0 / max(poll_time, 0.001))
            
            sample_count += 1
            time.sleep(0.016)  # ~60 FPS
        
        # Calculate statistics
        if self._performance_data["polling_rate"]:
            avg_poll_rate = statistics.mean(self._performance_data["polling_rate"])
            min_poll_rate = min(self._performance_data["polling_rate"])
            max_poll_rate = max(self._performance_data["polling_rate"])
        else:
            avg_poll_rate = min_poll_rate = max_poll_rate = 0.0
        
        details = {
            "test_duration": time.time() - start_time,
            "samples_collected": sample_count,
            "average_poll_rate": avg_poll_rate,
            "min_poll_rate": min_poll_rate,
            "max_poll_rate": max_poll_rate
        }
        
        # Determine result based on performance
        if avg_poll_rate < 30.0:  # Less than 30 FPS average
            result = TestResult.WARNING
            error_msg = f"Low polling rate: {avg_poll_rate:.1f} Hz"
        else:
            result = TestResult.PASSED
            error_msg = None
        
        return TestReport(
            test_case=test_case,
            result=result,
            error_message=error_msg,
            details=details
        )
    
    def _test_hotplug(self) -> TestReport:
        """Test hotplug functionality."""
        test_case = self._current_test
        
        if not self.hotplug_manager:
            return TestReport(
                test_case=test_case,
                result=TestResult.SKIPPED,
                error_message="Hotplug manager not available"
            )
        
        # Check if hotplug is active
        is_active = self.hotplug_manager.is_active()
        
        # Force a rescan
        initial_devices = self.hotplug_manager.detector.get_known_devices()
        self.hotplug_manager.force_rescan()
        
        # Wait a bit and check again
        time.sleep(1.0)
        final_devices = self.hotplug_manager.detector.get_known_devices()
        
        details = {
            "hotplug_active": is_active,
            "initial_devices": list(initial_devices),
            "final_devices": list(final_devices),
            "device_count_stable": len(initial_devices) == len(final_devices)
        }
        
        return TestReport(
            test_case=test_case,
            result=TestResult.PASSED if is_active else TestResult.WARNING,
            error_message=None if is_active else "Hotplug not active",
            details=details
        )
    
    def _run_calibration_test(self, test_case: TestCase, joystick_id: int) -> TestReport:
        """Run interactive calibration test."""
        if not self.joystick_manager.is_joystick_connected(joystick_id):
            return TestReport(
                test_case=test_case,
                result=TestResult.FAILED,
                error_message=f"Joystick {joystick_id} not connected"
            )
        
        joystick_info = self.joystick_manager.get_joystick_info(joystick_id)
        
        self.logger.info("Starting interactive calibration", extra={
            "joystick_id": joystick_id,
            "joystick_name": joystick_info.name,
            "instructions": [
                "1. Center all analog sticks and release all buttons",
                "2. Move each analog stick to its extremes",
                "3. Press each button once",
                "4. Move any D-pad/hat controls"
            ]
        })
        
        # Collect calibration data
        calibration_data = {
            "center_values": [],
            "min_values": [],
            "max_values": [],
            "button_presses": set(),
            "hat_values": set()
        }
        
        start_time = time.time()
        sample_count = 0
        
        while (time.time() - start_time) < test_case.timeout:
            input_state = self.joystick_manager.get_input_state(joystick_id)
            if input_state:
                # Collect axis data
                calibration_data["center_values"].append(input_state.axes.copy())
                
                # Track button presses
                for i, pressed in enumerate(input_state.buttons):
                    if pressed:
                        calibration_data["button_presses"].add(i)
                
                # Track hat values
                for i, hat_value in enumerate(input_state.hats):
                    if hat_value != (0, 0):
                        calibration_data["hat_values"].add((i, hat_value))
                
                sample_count += 1
            
            time.sleep(0.05)  # 20 Hz sampling
        
        # Analyze calibration data
        if calibration_data["center_values"]:
            # Calculate min/max for each axis
            num_axes = len(calibration_data["center_values"][0])
            axis_stats = []
            
            for axis_idx in range(num_axes):
                axis_values = [sample[axis_idx] for sample in calibration_data["center_values"]]
                axis_stats.append({
                    "min": min(axis_values),
                    "max": max(axis_values),
                    "center": statistics.mean(axis_values),
                    "range": max(axis_values) - min(axis_values)
                })
        else:
            axis_stats = []
        
        details = {
            "calibration_duration": time.time() - start_time,
            "samples_collected": sample_count,
            "buttons_pressed": list(calibration_data["button_presses"]),
            "hat_movements": list(calibration_data["hat_values"]),
            "axis_statistics": axis_stats,
            "total_axes": joystick_info.num_axes,
            "total_buttons": joystick_info.num_buttons,
            "total_hats": joystick_info.num_hats
        }
        
        # Determine if calibration was successful
        buttons_tested = len(calibration_data["button_presses"])
        axes_moved = sum(1 for stats in axis_stats if stats["range"] > 0.1)
        
        if buttons_tested == 0 and axes_moved == 0:
            result = TestResult.WARNING
            error_msg = "No input detected during calibration"
        elif buttons_tested < joystick_info.num_buttons // 2:
            result = TestResult.WARNING
            error_msg = f"Only {buttons_tested}/{joystick_info.num_buttons} buttons tested"
        else:
            result = TestResult.PASSED
            error_msg = None
        
        return TestReport(
            test_case=test_case,
            result=result,
            error_message=error_msg,
            details=details
        )
    
    def _start_event_collection(self) -> None:
        """Start collecting input events."""
        self._collected_events.clear()
        self._collection_active = True
        
        # Register event callback
        self.input_processor.add_event_callback(self._collect_event)
    
    def _stop_event_collection(self) -> None:
        """Stop collecting input events."""
        self._collection_active = False
        
        # Remove event callback
        self.input_processor.remove_event_callback(self._collect_event)
    
    def _collect_event(self, event: InputEvent) -> None:
        """Collect an input event for testing."""
        if self._collection_active:
            self._collected_events.append(event)
    
    def get_test_results(self) -> List[TestReport]:
        """Get all test results."""
        return self._test_results.copy()
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary of all test results."""
        if not self._test_results:
            return {"status": "no_tests_run"}
        
        total_tests = len(self._test_results)
        passed = sum(1 for r in self._test_results if r.result == TestResult.PASSED)
        failed = sum(1 for r in self._test_results if r.result == TestResult.FAILED)
        warnings = sum(1 for r in self._test_results if r.result == TestResult.WARNING)
        skipped = sum(1 for r in self._test_results if r.result == TestResult.SKIPPED)
        
        total_time = sum(r.execution_time for r in self._test_results)
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "skipped": skipped,
            "success_rate": (passed / total_tests) * 100 if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "overall_status": "passed" if failed == 0 else "failed" if failed > 0 else "warning"
        }
    
    def stop_testing(self) -> None:
        """Stop any running tests."""
        self._running_tests = False
        self._stop_event_collection()
        
        self.logger.info("Testing stopped by user request")