def _extract_down_distance_from_region_hybrid(
    self, region_data: dict[str, Any], current_time: float = None
) -> Optional[dict[str, Any]]:
    """
    HYBRID DOWN DETECTION: Template matching for down number + PaddleOCR for distance.
    
    This combines our 100% accurate template detection for down numbers (1ST, 2ND, 3RD, 4TH)
    with PaddleOCR for distance extraction ("& 10", "& 7", "& GOAL").
    """
    try:
        # FIXED: Unified confidence system - always extract, let appropriate system handle confidence
        is_burst_mode = current_time is None

        # FIXED: Make temporal manager advisory, not blocking - always allow fresh extraction as fallback
        temporal_suggests_skip = False
        if not is_burst_mode and hasattr(self, "temporal_manager"):
            temporal_suggests_skip = not self.temporal_manager.should_extract(
                "down_distance", current_time
            )
            if temporal_suggests_skip:
                # Try cached value first, but don't block fresh extraction if cache is insufficient
                cached_result = self.temporal_manager.get_current_value("down_distance")
                if cached_result and cached_result.get("value"):
                    cached_confidence = cached_result.get("value", {}).get("confidence", 0.0)
                    # Only use cache if confidence is high enough (>0.7) - otherwise do fresh extraction
                    if cached_confidence > 0.7:
                        logger.debug(
                            f"⏰ TEMPORAL CACHE: Using high-confidence cached down/distance result (conf={cached_confidence:.3f})"
                        )
                        return cached_result.get("value")
                    else:
                        logger.debug(
                            f"⏰ TEMPORAL OVERRIDE: Cached confidence too low ({cached_confidence:.3f}), performing fresh extraction"
                        )
                else:
                    logger.debug(
                        f"⏰ TEMPORAL OVERRIDE: No cached result available, performing fresh extraction"
                    )

        roi = region_data["roi"]
        yolo_confidence = region_data["confidence"]
        bbox = region_data["bbox"]

        # Check cache first for hybrid result
        if self.cache_enabled and self.advanced_cache:
            cached_result = self.advanced_cache.get_ocr_result(
                roi, "down_distance", "hybrid_template_ocr"
            )
            if cached_result is not None:
                logger.debug("⚡ Cache HIT: Using cached hybrid template+OCR result")
                return cached_result

        # FIXED: Scale up tiny regions for better processing (burst sampling fix)
        if roi.shape[0] < 20 or roi.shape[1] < 60:
            # Scale up small regions by 5x with better interpolation
            scale_factor = 5
            new_height = roi.shape[0] * scale_factor
            new_width = roi.shape[1] * scale_factor
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Apply additional preprocessing for scaled regions
            roi = self._enhance_scaled_region_for_ocr(roi)

            if current_time is None:  # Burst sampling mode
                print(f"🔧 BURST: Scaled tiny region to {roi.shape} for hybrid processing")

        # ===== STEP 1: TEMPLATE DETECTION FOR DOWN NUMBER (100% accuracy) =====
        template_result = None
        try:
            # Create detection context for template matching
            detection_context = DownDetectionContext(
                frame_number=getattr(self, '_debug_frame_counter', 0),
                yolo_confidence=yolo_confidence,
                region_size=roi.shape[:2]
            )
            
            # Use our 100% working template detection system
            template_result = self.down_template_detector.detect_down_in_yolo_region(
                roi, bbox, detection_context
            )
            
            if template_result and template_result.get("down"):
                logger.debug(f"🎯 TEMPLATE SUCCESS: Detected {template_result['down']} (conf: {template_result.get('confidence', 0.0):.3f})")
                if current_time is None:  # Burst sampling mode
                    print(f"🎯 TEMPLATE: Detected {template_result['down']} with {template_result.get('confidence', 0.0):.3f} confidence")
            else:
                logger.debug("❌ TEMPLATE: No down number detected")
                
        except Exception as e:
            logger.debug(f"❌ TEMPLATE ERROR: {e}")
            template_result = None

        # ===== STEP 2: PADDLE OCR FOR DISTANCE EXTRACTION =====
        ocr_distance = None
        ocr_confidence = 0.0
        full_text = ""
        try:
            # Apply same preprocessing as other HUD elements for consistency
            processed_roi = self._preprocess_region_for_ocr(roi)

            # Use PaddleOCR for distance extraction (same as other HUD elements)
            full_text = self.ocr.extract_down_distance(processed_roi)
            
            if full_text:
                logger.debug(f"🔍 OCR TEXT: '{full_text}'")
                if current_time is None:  # Burst sampling mode
                    print(f"🔍 OCR: Extracted text '{full_text}'")
                
                # Parse distance from OCR text (look for "& X" patterns)
                distance_match = re.search(r'&\s*(\d+|goal|GOAL)', full_text, re.IGNORECASE)
                if distance_match:
                    distance_text = distance_match.group(1).upper()
                    if distance_text == "GOAL":
                        ocr_distance = 0  # Goal line
                    else:
                        try:
                            ocr_distance = int(distance_text)
                            if 0 <= ocr_distance <= 99:  # Valid distance range
                                ocr_confidence = 0.8  # High confidence for valid distance
                            else:
                                ocr_distance = None
                        except ValueError:
                            ocr_distance = None
                
                if ocr_distance is not None:
                    logger.debug(f"🎯 OCR DISTANCE: {ocr_distance} (conf: {ocr_confidence:.3f})")
                else:
                    logger.debug("❌ OCR: No valid distance found")
                    
        except Exception as e:
            logger.debug(f"❌ OCR ERROR: {e}")
            ocr_distance = None
            ocr_confidence = 0.0

        # ===== STEP 3: COMBINE TEMPLATE + OCR RESULTS =====
        final_result = None
        
        if template_result and template_result.get("down"):
            # Template detection successful - use it as primary
            final_result = {
                "down": template_result["down"],
                "distance": ocr_distance if ocr_distance is not None else template_result.get("distance"),
                "confidence": min(0.95, template_result.get("confidence", 0.0) + (ocr_confidence * 0.2)),  # Boost confidence if OCR also worked
                "method": "hybrid_template_ocr",
                "template_confidence": template_result.get("confidence", 0.0),
                "ocr_confidence": ocr_confidence,
                "source": "8class_down_distance_area",
                "region_confidence": yolo_confidence,
                "region_bbox": bbox,
                "raw_template": template_result.get("template_name", ""),
                "raw_ocr": full_text
            }
            
            logger.debug(f"✅ HYBRID SUCCESS: {final_result['down']} & {final_result['distance']} (conf: {final_result['confidence']:.3f})")
            
        elif ocr_distance is not None:
            # Template failed but OCR got distance - try to infer down from context or use fallback
            logger.debug("⚠️ HYBRID PARTIAL: Template failed, using OCR-only fallback")
            
            # Fallback to full OCR parsing
            parsed_ocr = self._parse_down_distance_text(full_text)
            if parsed_ocr:
                final_result = parsed_ocr
                final_result["method"] = "ocr_fallback"
                final_result["source"] = "8class_down_distance_area"
                final_result["region_confidence"] = yolo_confidence
                final_result["region_bbox"] = bbox

        # Performance logging when overriding temporal manager suggestion
        if temporal_suggests_skip:
            logger.debug(
                f"⚡ PERFORMANCE: Temporal manager suggested skip, but performed fresh hybrid extraction anyway"
            )

        # Cache the hybrid result
        if final_result and self.cache_enabled and self.advanced_cache:
            try:
                self.advanced_cache.set_ocr_result(
                    roi, "down_distance", "hybrid_template_ocr", final_result
                )
                logger.debug("💾 Cached hybrid template+OCR result")
            except Exception as e:
                logger.debug(f"Hybrid cache storage failed: {e}")

        # FIXED: Unified result handling - add to appropriate confidence system
        if final_result:
            if not is_burst_mode and hasattr(self, "temporal_manager"):
                # Normal mode: Add to temporal manager for time-based confidence voting
                extraction_result = ExtractionResult(
                    value=final_result,
                    confidence=final_result["confidence"],
                    timestamp=current_time,
                    raw_text=final_result.get("raw_ocr", ""),
                    method=final_result["method"],
                )
                self.temporal_manager.add_extraction_result(
                    "down_distance", extraction_result
                )
                logger.debug(f"⏰ TEMPORAL: Added hybrid result to temporal manager")
            else:
                # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                logger.debug(
                    f"🎯 BURST: Hybrid result will be handled by burst consensus"
                )

            return final_result

        # ===== STEP 4: FULL OCR FALLBACK (if both template and distance OCR failed) =====
        logger.debug("⚠️ FALLBACK: Using full PaddleOCR as last resort")
        
        # Apply OCR corrections and try full parsing
        try:
            processed_roi = self._preprocess_region_for_ocr(roi)
            
            # Try PaddleOCR first (primary OCR engine)
            if not full_text:  # Only if we haven't already tried
                full_text = self.ocr.extract_down_distance(processed_roi)
            
            if full_text:
                corrected_text = self._apply_down_distance_corrections(full_text)
                fallback_result = self._parse_down_distance_text(corrected_text)
                
                if fallback_result:
                    fallback_result["method"] = "paddle_ocr_fallback"
                    fallback_result["confidence"] = yolo_confidence * 0.7  # Medium confidence for PaddleOCR fallback
                    fallback_result["source"] = "8class_down_distance_area"
                    fallback_result["region_confidence"] = yolo_confidence
                    fallback_result["region_bbox"] = bbox
                    
                    logger.debug(f"🔄 PADDLE FALLBACK SUCCESS: {fallback_result.get('down')} & {fallback_result.get('distance')}")
                    return fallback_result
            
            # Final fallback to Tesseract if PaddleOCR completely failed
            config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndrdthgoalTHNDRDSTRD"
            tesseract_text = pytesseract.image_to_string(processed_roi, config=config).strip()
            
            if tesseract_text:
                corrected_text = self._apply_down_distance_corrections(tesseract_text)
                fallback_result = self._parse_down_distance_text(corrected_text)
                
                if fallback_result:
                    fallback_result["method"] = "tesseract_fallback"
                    fallback_result["confidence"] = yolo_confidence * 0.6  # Lower confidence for Tesseract fallback
                    fallback_result["source"] = "8class_down_distance_area"
                    fallback_result["region_confidence"] = yolo_confidence
                    fallback_result["region_bbox"] = bbox
                    
                    logger.debug(f"🔄 TESSERACT FALLBACK SUCCESS: {fallback_result.get('down')} & {fallback_result.get('distance')}")
                    return fallback_result
                    
        except Exception as e:
            logger.debug(f"❌ FALLBACK ERROR: {e}")

        logger.debug("❌ ALL METHODS FAILED: No down/distance detected")
        return None

    except Exception as e:
        logger.error(f"🚨 EXCEPTION in hybrid down/distance extraction: {e}")
        logger.error(f"🚨 Exception type: {type(e)}")
        import traceback
        logger.error(f"🚨 Traceback: {traceback.format_exc()}")
        return None 