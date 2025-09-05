# Update the WebSocket handling section in main.py
@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """FIXED: WebSocket endpoint for conversational AI with proper mistral-common processing"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 FIXED UNDERSTANDING session started with mistral-common")
        
        while not shutdown_event.is_set():
            try:
                # Receive binary audio data
                audio_data = await websocket.receive_bytes()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Validate audio data
                if not audio_data or len(audio_data) < 100:
                    logger.debug("Invalid/insufficient audio data received")
                    continue
                
                # FIXED: Process through corrected audio processor
                result = await audio_processor.process_audio_understanding(audio_data, websocket)
                
                if result and isinstance(result, dict):
                    if "error" in result:
                        logger.error(f"FIXED audio processing error: {result['error']}")
                        await websocket.send_json({
                            "error": f"Audio processing failed: {result['error']}", 
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                        continue
                    
                    # Send intermediate feedback
                    if result.get("audio_received") and not result.get("speech_complete"):
                        await websocket.send_json({
                            "type": "audio_feedback",
                            "audio_received": True,
                            "segment_duration_ms": result.get("segment_duration_ms", 0),
                            "silence_duration_ms": result.get("silence_duration_ms", 0),
                            "remaining_to_gap_ms": result.get("remaining_to_gap_ms", 0),
                            "gap_will_trigger_at_ms": result.get("gap_will_trigger_at_ms", 300),
                            "speech_detected": result.get("speech_detected", False),
                            "speech_ratio": result.get("speech_ratio", 0),
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                        continue
                    
                    # FIXED: Process complete speech segment with mistral-common
                    if result.get("speech_complete") and "audio_data" in result:
                        duration_ms = result.get("duration_ms", 0)
                        speech_quality = result.get("speech_quality", 0)
                        
                        # Quality check for understanding
                        if duration_ms > 500 and speech_quality > 0.15:  # Lowered quality threshold
                            logger.info(f"🧠 FIXED processing with mistral-common: {duration_ms:.0f}ms, quality: {speech_quality:.3f}")
                            
                            if model_manager and model_manager.is_loaded:
                                # Get conversation context
                                context = conversation_manager.get_conversation_context(websocket)
                                
                                # CRITICAL: Create proper message format for mistral-common
                                message = {
                                    "audio": result["audio_data"],  # Raw PCM data
                                    "text": f"Please understand and respond to what you hear in the audio. {context}" if context else "Please understand and respond to what you hear in the audio."
                                }
                                
                                # FIXED: Generate understanding response using correct mistral-common API
                                understanding_result = await model_manager.understand_audio(message)
                                
                                if (isinstance(understanding_result, dict) and 
                                    understanding_result.get("response") and 
                                    "error" not in understanding_result and
                                    len(understanding_result["response"].strip()) > 3):
                                    
                                    response_time_ms = understanding_result.get("processing_time_ms", 0)
                                    transcribed_text = understanding_result.get("transcribed_text", "Audio processed with mistral-common")
                                    
                                    # Add to conversation
                                    conversation_manager.add_turn(
                                        websocket,
                                        transcription=transcribed_text,
                                        response=understanding_result["response"],
                                        audio_duration=duration_ms / 1000,
                                        speech_ratio=speech_quality,
                                        mode="understand",
                                        language=understanding_result.get("language", "en")
                                    )
                                    
                                    # Prepare response
                                    final_result = {
                                        "type": "understanding",
                                        "transcription": transcribed_text,
                                        "response": understanding_result["response"],
                                        "response_time_ms": response_time_ms,
                                        "audio_duration_ms": duration_ms,
                                        "speech_quality": speech_quality,
                                        "gap_detected": result.get("gap_detected", False),
                                        "language": understanding_result.get("language", "en"),
                                        "understanding_only": True,
                                        "transcription_disabled": True,
                                        "mistral_common_integrated": True,
                                        "fixes_applied": True,
                                        "model_api_fixed": understanding_result.get("model_api_fixed", False),
                                        "sub_200ms": response_time_ms < 200,
                                        "timestamp": asyncio.get_event_loop().time()
                                    }
                                    
                                    # Add conversation stats
                                    conv_stats = conversation_manager.get_conversation_stats(websocket)
                                    final_result["conversation"] = conv_stats
                                    
                                    await websocket.send_json(final_result)
                                    logger.info(f"✅ FIXED UNDERSTANDING complete with mistral-common: '{understanding_result['response'][:50]}...' ({response_time_ms:.0f}ms)")
                                else:
                                    logger.warning(f"Invalid understanding result: {understanding_result}")
                                    await websocket.send_json({
                                        "error": "Failed to generate understanding response",
                                        "understanding_only": True,
                                        "transcription_disabled": True,
                                        "fixes_applied": True
                                    })
                            else:
                                await websocket.send_json({
                                    "error": "Model not loaded",
                                    "understanding_only": True,
                                    "transcription_disabled": True,
                                    "fixes_applied": True
                                })
                        else:
                            logger.debug(f"Skipping low quality: duration={duration_ms:.0f}ms, quality={speech_quality:.3f}")
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"FIXED inner WebSocket error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("FIXED WebSocket disconnected")
    except Exception as e:
        logger.error(f"FIXED WebSocket error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        audio_processor.cleanup_connection(websocket)
        ws_manager.disconnect(websocket)
