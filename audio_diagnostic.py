#!/usr/bin/env python3
"""
COMPLETE DIAGNOSTIC TOOL for Voxtral Audio Processing Issues
"""

import subprocess
import tempfile
import os
import logging

def test_ffmpeg_installation():
    """Test if FFmpeg is properly installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and accessible")
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
            return True
        else:
            print("❌ FFmpeg is installed but returned error")
            print(f"   Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing FFmpeg: {e}")
        return False

def test_webm_support():
    """Test if FFmpeg supports WebM format"""
    try:
        # Test WebM demuxer
        result = subprocess.run(['ffmpeg', '-f', 'webm', '-i', '/dev/null'], 
                              capture_output=True, text=True, timeout=5)
        # This will fail (no input), but should mention WebM support
        if 'webm' in result.stderr.lower() or 'matroska' in result.stderr.lower():
            print("✅ WebM format support detected")
            return True
        else:
            print("❌ WebM format support not clearly detected")
            print(f"   FFmpeg output: {result.stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Error testing WebM support: {e}")
        return False

def test_opus_codec():
    """Test if FFmpeg supports Opus codec"""
    try:
        result = subprocess.run(['ffmpeg', '-codecs'], capture_output=True, text=True, timeout=10)
        if 'opus' in result.stdout.lower():
            print("✅ Opus codec support detected")
            return True
        else:
            print("❌ Opus codec support not found")
            return False
    except Exception as e:
        print(f"❌ Error testing Opus codec: {e}")
        return False

def test_audio_conversion():
    """Test actual audio conversion with a synthetic WebM file"""
    try:
        print("🔧 Testing audio conversion with synthetic data...")
        
        # Create a temporary WebM file with FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            webm_path = temp_webm.name
        
        # Generate 1 second of test audio in WebM format
        cmd_generate = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
            '-c:a', 'libopus', '-f', 'webm', webm_path
        ]
        
        result = subprocess.run(cmd_generate, capture_output=True, timeout=15)
        if result.returncode != 0:
            print("❌ Failed to generate test WebM file")
            print(f"   Error: {result.stderr.decode()}")
            return False
        
        print("✅ Generated test WebM file")
        
        # Now try to convert it to PCM
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
            pcm_path = temp_pcm.name
        
        cmd_convert = [
            'ffmpeg', '-y', '-i', webm_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-f', 's16le', pcm_path
        ]
        
        result = subprocess.run(cmd_convert, capture_output=True, timeout=15)
        
        if result.returncode == 0 and os.path.exists(pcm_path) and os.path.getsize(pcm_path) > 0:
            print("✅ Successfully converted WebM to PCM")
            pcm_size = os.path.getsize(pcm_path)
            print(f"   PCM file size: {pcm_size} bytes")
            
            # Clean up
            os.unlink(webm_path)
            os.unlink(pcm_path)
            return True
        else:
            print("❌ Failed to convert WebM to PCM")
            print(f"   Error: {result.stderr.decode()}")
            
            # Clean up
            if os.path.exists(webm_path):
                os.unlink(webm_path)
            if os.path.exists(pcm_path):
                os.unlink(pcm_path)
            return False
            
    except Exception as e:
        print(f"❌ Error in audio conversion test: {e}")
        return False

def test_python_dependencies():
    """Test if required Python packages are available"""
    required_packages = [
        'numpy', 'webrtcvad', 'asyncio', 'tempfile', 'subprocess'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is NOT available")
            all_good = False
    
    return all_good

def main():
    """Run complete diagnostic"""
    print("=" * 60)
    print("🔍 VOXTRAL AUDIO PROCESSING DIAGNOSTIC")
    print("=" * 60)
    
    print("\n1. Testing FFmpeg installation...")
    ffmpeg_ok = test_ffmpeg_installation()
    
    print("\n2. Testing WebM format support...")
    webm_ok = test_webm_support()
    
    print("\n3. Testing Opus codec support...")
    opus_ok = test_opus_codec()
    
    print("\n4. Testing Python dependencies...")
    python_ok = test_python_dependencies()
    
    print("\n5. Testing actual audio conversion...")
    conversion_ok = test_audio_conversion()
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if all([ffmpeg_ok, webm_ok, opus_ok, python_ok, conversion_ok]):
        print("🎉 ALL TESTS PASSED! Audio processing should work correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Issues detected:")
        if not ffmpeg_ok:
            print("   - FFmpeg installation issues")
        if not webm_ok:
            print("   - WebM format support issues") 
        if not opus_ok:
            print("   - Opus codec support issues")
        if not python_ok:
            print("   - Python dependency issues")
        if not conversion_ok:
            print("   - Audio conversion pipeline issues")
        
        print("\n🔧 RECOMMENDED FIXES:")
        if not ffmpeg_ok:
            print("   - Install FFmpeg: apt-get update && apt-get install -y ffmpeg")
        if not webm_ok or not opus_ok:
            print("   - Install FFmpeg with WebM/Opus support")
        if not python_ok:
            print("   - Install missing Python packages: pip install numpy webrtcvad")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
