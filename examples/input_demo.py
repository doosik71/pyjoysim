#!/usr/bin/env python3
"""
PyJoySim Input System Demo

This script demonstrates the complete joystick input system including:
- Joystick detection and management
- Input processing and event handling
- Hot-plug support
- Configuration management
- Testing and validation

Usage:
    python examples/input_demo.py [--test] [--joystick-id ID] [--profile PROFILE]
"""

import sys
import time
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyjoysim.input import (
    get_joystick_manager,
    get_input_processor,
    get_input_config_manager,
    initialize_hotplug_manager,
    InputTester,
    InputEvent,
    HotplugEvent,
    SmoothingFilter,
    DeadzonFilter
)
from pyjoysim.core.logging import get_logger
from pyjoysim.config import get_config


class InputDemo:
    """Interactive demo of the PyJoySim input system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = get_logger("input_demo")
        
        # Initialize managers
        self.joystick_manager = get_joystick_manager()
        self.input_processor = get_input_processor()
        self.config_manager = get_input_config_manager()
        self.hotplug_manager = None
        self.tester = None
        
        # Demo state
        self.running = False
        self.event_count = 0
        self.last_event_time = 0.0
        
    def initialize(self) -> bool:
        """Initialize all components."""
        self.logger.info("Initializing PyJoySim Input Demo")
        
        # Initialize joystick manager
        if not self.joystick_manager.initialize():
            self.logger.error("Failed to initialize joystick manager")
            return False
        
        # Initialize hotplug support
        self.hotplug_manager = initialize_hotplug_manager(self.joystick_manager)
        if not self.hotplug_manager.start():
            self.logger.warning("Failed to start hotplug manager")
        
        # Initialize tester
        self.tester = InputTester(
            self.joystick_manager,
            self.input_processor,
            self.config_manager,
            self.hotplug_manager
        )
        
        # Register event callbacks
        self.input_processor.add_event_callback(self._on_input_event)
        if self.hotplug_manager:
            self.hotplug_manager.detector.add_callback(self._on_hotplug_event)
        
        # Add some input filters for demonstration
        smoothing_filter = SmoothingFilter(smoothing_factor=0.3)
        deadzone_filter = DeadzonFilter({0: 0.1, 1: 0.1})  # Apply to first two axes
        
        self.input_processor.add_filter(smoothing_filter)
        self.input_processor.add_filter(deadzone_filter)
        
        self.logger.info("Demo initialization complete")
        return True
    
    def shutdown(self):
        """Shutdown all components."""
        self.logger.info("Shutting down demo")
        
        self.running = False
        
        if self.hotplug_manager:
            self.hotplug_manager.stop()
        
        self.joystick_manager.shutdown()
        
        self.logger.info("Demo shutdown complete")
    
    def run_interactive_demo(self):
        """Run interactive demonstration."""
        self.logger.info("Starting interactive demo")
        print("\n" + "="*60)
        print("PyJoySim Input System Interactive Demo")
        print("="*60)
        
        # Show initial status
        self._show_system_status()
        
        # Main demo loop
        self.running = True
        self._show_menu()
        
        try:
            while self.running:
                # Update systems
                self.joystick_manager.update()
                
                # Process any joystick input
                for joystick_id in self.joystick_manager.get_all_joysticks().keys():
                    input_state = self.joystick_manager.get_input_state(joystick_id)
                    if input_state:
                        events = self.input_processor.process_input(joystick_id, input_state)
                        # Events are handled by callbacks
                
                # Check for user commands
                self._check_user_input()
                
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        self.shutdown()
    
    def run_automated_tests(self, joystick_id: int = None):
        """Run automated test suite."""
        self.logger.info("Running automated tests", extra={"joystick_id": joystick_id})
        
        print("\n" + "="*60)
        print("PyJoySim Input System Automated Tests")
        print("="*60)
        
        # Run basic tests
        print("\nRunning basic tests...")
        basic_results = self.tester.run_basic_tests(joystick_id)
        self._print_test_results("Basic Tests", basic_results)
        
        # Run performance tests
        print("\nRunning performance tests (30 seconds)...")
        perf_results = self.tester.run_performance_tests(30.0, joystick_id)
        self._print_test_results("Performance Tests", perf_results)
        
        # Show summary
        summary = self.tester.get_summary_report()
        self._print_test_summary(summary)
        
        return summary["overall_status"] == "passed"
    
    def _show_system_status(self):
        """Show current system status."""
        print(f"\nSystem Status:")
        print(f"  Joystick Manager: {'✓ Initialized' if self.joystick_manager._initialized else '✗ Not initialized'}")
        print(f"  Hotplug Manager: {'✓ Active' if self.hotplug_manager and self.hotplug_manager.is_active() else '✗ Inactive'}")
        print(f"  Connected Joysticks: {self.joystick_manager.get_joystick_count()}")
        
        # Show joystick details
        joysticks = self.joystick_manager.get_all_joysticks()
        for joystick_id, info in joysticks.items():
            print(f"    Joystick {joystick_id}: {info.name}")
            print(f"      GUID: {info.guid}")
            print(f"      Axes: {info.num_axes}, Buttons: {info.num_buttons}, Hats: {info.num_hats}")
        
        # Show available profiles
        profiles = self.config_manager.get_all_profiles()
        print(f"  Available Profiles: {len(profiles)}")
        for profile_id, profile in profiles.items():
            print(f"    {profile_id}: {profile.name} ({profile.profile_type.value})")
    
    def _show_menu(self):
        """Show interactive menu."""
        print("\nInteractive Demo Menu:")
        print("  [s] Show system status")
        print("  [j] List joysticks")
        print("  [p] List profiles")
        print("  [t] Run basic tests")
        print("  [c] Run calibration (interactive)")
        print("  [m] Monitor input (real-time)")
        print("  [h] Toggle hotplug detection")
        print("  [q] Quit")
        print("\nMove joysticks or press buttons to see input events...")
        print("Press any key for menu commands...")
    
    def _check_user_input(self):
        """Check for user keyboard input (non-blocking)."""
        # This is a simplified version - in a real application you'd use
        # proper non-blocking input or a GUI framework
        import select
        import sys
        
        if select.select([sys.stdin], [], [], 0.0)[0]:
            try:
                command = sys.stdin.readline().strip().lower()
                self._handle_command(command)
            except:
                pass  # Ignore input errors
    
    def _handle_command(self, command: str):
        """Handle user commands."""
        if command == 'q':
            self.running = False
        elif command == 's':
            self._show_system_status()
        elif command == 'j':
            self._list_joysticks()
        elif command == 'p':
            self._list_profiles()
        elif command == 't':
            self._run_quick_tests()
        elif command == 'c':
            self._run_calibration()
        elif command == 'm':
            self._monitor_input()
        elif command == 'h':
            self._toggle_hotplug()
        else:
            print(f"Unknown command: {command}")
            self._show_menu()
    
    def _list_joysticks(self):
        """List all joysticks with detailed info."""
        joysticks = self.joystick_manager.get_all_joysticks()
        print(f"\nConnected Joysticks ({len(joysticks)}):")
        
        for joystick_id, info in joysticks.items():
            state = self.joystick_manager.get_input_state(joystick_id)
            print(f"  Joystick {joystick_id}: {info.name}")
            print(f"    State: {info.state.value}")
            print(f"    Hardware: {info.num_axes} axes, {info.num_buttons} buttons, {info.num_hats} hats")
            
            if state:
                print(f"    Current Input:")
                print(f"      Axes: {[f'{v:.3f}' for v in state.axes]}")
                print(f"      Buttons: {[i for i, pressed in enumerate(state.buttons) if pressed]}")
                print(f"      Hats: {state.hats}")
            
            # Show assigned profile
            profile = self.config_manager.get_profile_for_joystick(info.guid)
            if profile:
                print(f"    Profile: {profile.name} ({profile.profile_type.value})")
    
    def _list_profiles(self):
        """List all available profiles."""
        profiles = self.config_manager.get_all_profiles()
        print(f"\nAvailable Profiles ({len(profiles)}):")
        
        for profile_id, profile in profiles.items():
            print(f"  {profile_id}: {profile.name}")
            print(f"    Type: {profile.profile_type.value}")
            print(f"    Description: {profile.description}")
            print(f"    Deadzone: {profile.deadzone}, Sensitivity: {profile.sensitivity}")
            
            # Validate profile
            errors = self.config_manager.validate_profile(profile)
            if errors:
                print(f"    ⚠ Validation Errors: {len(errors)}")
            else:
                print(f"    ✓ Valid Profile")
    
    def _run_quick_tests(self):
        """Run quick basic tests."""
        print("\nRunning quick tests...")
        results = self.tester.run_basic_tests()
        self._print_test_results("Quick Tests", results)
    
    def _run_calibration(self):
        """Run interactive calibration."""
        joysticks = self.joystick_manager.get_all_joysticks()
        if not joysticks:
            print("No joysticks connected for calibration")
            return
        
        # Use first joystick
        joystick_id = list(joysticks.keys())[0]
        print(f"\nRunning calibration for Joystick {joystick_id}...")
        print("Follow the on-screen instructions...")
        
        result = self.tester.run_interactive_calibration(joystick_id)
        self._print_test_results("Calibration", [result])
    
    def _monitor_input(self):
        """Monitor real-time input for a few seconds."""
        print("\nMonitoring input for 10 seconds...")
        print("Move joysticks to see real-time input data...")
        
        start_time = time.time()
        while (time.time() - start_time) < 10.0:
            for joystick_id in self.joystick_manager.get_all_joysticks().keys():
                state = self.joystick_manager.get_input_state(joystick_id)
                if state and any(abs(v) > 0.1 for v in state.axes):
                    print(f"  JS{joystick_id}: Axes={[f'{v:.2f}' for v in state.axes[:4]]}")
            
            time.sleep(0.1)
        
        print("Monitoring complete.")
    
    def _toggle_hotplug(self):
        """Toggle hotplug detection."""
        if not self.hotplug_manager:
            print("Hotplug manager not available")
            return
        
        if self.hotplug_manager.is_active():
            self.hotplug_manager.stop()
            print("Hotplug detection stopped")
        else:
            if self.hotplug_manager.start():
                print("Hotplug detection started")
            else:
                print("Failed to start hotplug detection")
    
    def _on_input_event(self, event: InputEvent):
        """Handle input events."""
        self.event_count += 1
        current_time = time.time()
        
        # Throttle event display (max 10 per second)
        if (current_time - self.last_event_time) >= 0.1:
            print(f"Input Event: {event}")
            self.last_event_time = current_time
    
    def _on_hotplug_event(self, event: HotplugEvent):
        """Handle hotplug events."""
        print(f"Hotplug Event: {event.event_type.value} - Joystick {event.joystick_id}")
        if event.device_info:
            name = event.device_info.get('name', 'Unknown')
            print(f"  Device: {name}")
    
    def _print_test_results(self, test_suite: str, results: list):
        """Print formatted test results."""
        print(f"\n{test_suite} Results:")
        print("-" * 40)
        
        for result in results:
            status_icon = {
                "passed": "✓",
                "failed": "✗",
                "warning": "⚠",
                "skipped": "○"
            }.get(result.result.value, "?")
            
            print(f"{status_icon} {result.test_case.name}")
            if result.error_message:
                print(f"    Error: {result.error_message}")
            if result.execution_time > 0:
                print(f"    Time: {result.execution_time:.2f}s")
    
    def _print_test_summary(self, summary: dict):
        """Print test summary."""
        print("\nTest Summary:")
        print("-" * 40)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {summary['total_execution_time']:.2f}s")
        print(f"Overall Status: {summary['overall_status'].upper()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyJoySim Input System Demo")
    parser.add_argument("--test", action="store_true", help="Run automated tests instead of interactive demo")
    parser.add_argument("--joystick-id", type=int, help="Specific joystick ID to test")
    parser.add_argument("--profile", help="Profile to apply during demo")
    
    args = parser.parse_args()
    
    # Create and initialize demo
    demo = InputDemo()
    
    if not demo.initialize():
        print("Failed to initialize demo")
        return 1
    
    try:
        if args.test:
            # Run automated tests
            success = demo.run_automated_tests(args.joystick_id)
            return 0 if success else 1
        else:
            # Run interactive demo
            demo.run_interactive_demo()
            return 0
    
    except Exception as e:
        print(f"Demo error: {e}")
        demo.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())