"""Minimal, read-only IOKit inventory for descriptor-bound serial identity."""

from __future__ import annotations

import ctypes
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

from .serial_port_identity import MacOSSerialRegistryCandidate, _binding_digest_components

_KERN_SUCCESS = 0
_IO_SERVICE_PLANE = b"IOService"
_IO_NAME_SIZE = 128
_CF_STRING_ENCODING_UTF8 = 0x08000100
_CF_NUMBER_SINT16_TYPE = 2


class _IOKitBindings:
    """Typed ctypes bindings, created only on macOS."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise OSError("IOKit serial inventory is available only on macOS")
        self.iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        io_object = ctypes.c_uint32
        io_object_pointer = ctypes.POINTER(io_object)

        self.iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        self.iokit.IOServiceMatching.restype = ctypes.c_void_p
        self.iokit.IOServiceGetMatchingServices.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            io_object_pointer,
        ]
        self.iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
        self.iokit.IOIteratorNext.argtypes = [io_object]
        self.iokit.IOIteratorNext.restype = io_object
        self.iokit.IOIteratorIsValid.argtypes = [io_object]
        self.iokit.IOIteratorIsValid.restype = ctypes.c_bool
        self.iokit.IOObjectGetClass.argtypes = [io_object, ctypes.c_void_p]
        self.iokit.IOObjectGetClass.restype = ctypes.c_int
        self.iokit.IOObjectRelease.argtypes = [io_object]
        self.iokit.IOObjectRelease.restype = ctypes.c_int
        self.iokit.IORegistryEntryGetParentEntry.argtypes = [
            io_object,
            ctypes.c_char_p,
            io_object_pointer,
        ]
        self.iokit.IORegistryEntryGetParentEntry.restype = ctypes.c_int
        self.iokit.IORegistryEntryGetRegistryEntryID.argtypes = [
            io_object,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.iokit.IORegistryEntryGetRegistryEntryID.restype = ctypes.c_int
        self.iokit.IORegistryEntryCreateCFProperty.argtypes = [
            io_object,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p

        self.cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        self.cf.CFStringGetCString.restype = ctypes.c_bool
        self.cf.CFNumberGetValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.cf.CFNumberGetValue.restype = ctypes.c_bool
        self.cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
        self.cf.CFGetTypeID.restype = ctypes.c_ulong
        self.cf.CFStringGetTypeID.argtypes = []
        self.cf.CFStringGetTypeID.restype = ctypes.c_ulong
        self.cf.CFNumberGetTypeID.argtypes = []
        self.cf.CFNumberGetTypeID.restype = ctypes.c_ulong
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]
        self.cf.CFRelease.restype = None

    def release_io_object(self, io_object: int) -> None:
        if io_object:
            self.iokit.IOObjectRelease(io_object)

    def registry_id(self, io_object: int) -> int | None:
        output = ctypes.c_uint64()
        if self.iokit.IORegistryEntryGetRegistryEntryID(io_object, ctypes.byref(output)) != _KERN_SUCCESS:
            return None
        return output.value

    def object_class(self, io_object: int) -> bytes | None:
        output = ctypes.create_string_buffer(_IO_NAME_SIZE)
        if self.iokit.IOObjectGetClass(io_object, ctypes.byref(output)) != _KERN_SUCCESS:
            return None
        return output.value

    def property(self, io_object: int, key_text: str) -> int | None:
        key = self.cf.CFStringCreateWithCString(
            None,
            key_text.encode("utf-8"),
            _CF_STRING_ENCODING_UTF8,
        )
        if not key:
            return None
        try:
            return self.iokit.IORegistryEntryCreateCFProperty(io_object, key, None, 0) or None
        finally:
            self.cf.CFRelease(key)

    def string_property(self, io_object: int, key_text: str) -> str | None:
        value = self.property(io_object, key_text)
        if value is None:
            return None
        try:
            if self.cf.CFGetTypeID(value) != self.cf.CFStringGetTypeID():
                return None
            output = ctypes.create_string_buffer(4096)
            if not self.cf.CFStringGetCString(
                value,
                ctypes.byref(output),
                len(output),
                _CF_STRING_ENCODING_UTF8,
            ):
                return None
            return output.value.decode("utf-8")
        except UnicodeError:
            return None
        finally:
            self.cf.CFRelease(value)

    def uint16_property(self, io_object: int, key_text: str) -> int | None:
        value = self.property(io_object, key_text)
        if value is None:
            return None
        try:
            if self.cf.CFGetTypeID(value) != self.cf.CFNumberGetTypeID():
                return None
            output = ctypes.c_uint16()
            if not self.cf.CFNumberGetValue(
                value,
                _CF_NUMBER_SINT16_TYPE,
                ctypes.byref(output),
            ):
                return None
            return output.value
        finally:
            self.cf.CFRelease(value)

    def services(self, service_class: bytes) -> Iterator[int]:
        iterator = ctypes.c_uint32()
        matching = self.iokit.IOServiceMatching(service_class)
        if not matching:
            raise OSError("IOKit could not create a service matching dictionary")
        result = self.iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator))
        if result != _KERN_SUCCESS:
            raise OSError("IOKit could not enumerate serial services")
        try:
            while True:
                service = self.iokit.IOIteratorNext(iterator.value)
                if not service:
                    if not self.iokit.IOIteratorIsValid(iterator.value):
                        raise OSError("IOKit serial inventory changed during enumeration")
                    break
                yield service
        finally:
            self.release_io_object(iterator.value)

    def usb_parent(self, service: int) -> int | None:
        for target_class in (b"IOUSBHostDevice", b"IOUSBDevice"):
            parent = self._parent_by_class(service, target_class)
            if parent is not None:
                return parent
        return None

    def _parent_by_class(self, service: int, target_class: bytes) -> int | None:
        current = service
        owns_current = False
        while True:
            if self.object_class(current) == target_class:
                return current if owns_current else None
            parent = ctypes.c_uint32()
            result = self.iokit.IORegistryEntryGetParentEntry(
                current,
                _IO_SERVICE_PLANE,
                ctypes.byref(parent),
            )
            if owns_current:
                self.release_io_object(current)
            if result != _KERN_SUCCESS or not parent.value:
                return None
            current = parent.value
            owns_current = True


def _candidate(bindings: _IOKitBindings, service: int) -> MacOSSerialRegistryCandidate | None:
    callout_device = bindings.string_property(service, "IOCalloutDevice")
    if not callout_device or not Path(callout_device).is_absolute():
        return None
    try:
        device_stat = os.stat(callout_device, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISCHR(device_stat.st_mode):
        return None

    usb_device = bindings.usb_parent(service)
    if usb_device is None:
        return None
    try:
        serial_registry_id = bindings.registry_id(service)
        usb_registry_id = bindings.registry_id(usb_device)
        binding = _binding_digest_components(
            bindings.string_property(usb_device, "USB Serial Number"),
            bindings.uint16_property(usb_device, "idVendor"),
            bindings.uint16_property(usb_device, "idProduct"),
        )
    finally:
        bindings.release_io_object(usb_device)
    if serial_registry_id is None or usb_registry_id is None or binding is None:
        return None
    return MacOSSerialRegistryCandidate(
        device_number=device_stat.st_rdev,
        serial_registry_id=serial_registry_id,
        usb_registry_id=usb_registry_id,
        binding=binding,
    )


def enumerate_serial_registry() -> tuple[MacOSSerialRegistryCandidate, ...]:
    """Return a complete USB-backed IOKit serial inventory or raise."""
    bindings = _IOKitBindings()
    candidates: list[MacOSSerialRegistryCandidate] = []
    for service in bindings.services(b"IOSerialBSDClient"):
        try:
            candidate = _candidate(bindings, service)
            if candidate is not None:
                candidates.append(candidate)
        finally:
            bindings.release_io_object(service)
    return tuple(candidates)
