"""
Prediction Client - Integrates ML predictions into Proactive Engine

This module calls the Prediction Service to get forecasts for devices,
enabling the Proactive Engine to make smarter, timing-aware suggestions.

Example Use Cases:
- "Light will turn on at 18:00 - suggest automation at 17:55"
- "HVAC usage predicted to spike - pre-cool before peak rates"
- "Motion sensor inactive next 8 hours - disable automations"
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)

PREDICTION_SERVICE_URL = "http://prediction_service:8000"


async def get_prediction(entity_id: str, hours_ahead: int = 24) -> Optional[Dict[str, Any]]:
    """
    Get prediction for an entity.
    
    Args:
        entity_id: Device to get prediction for
        hours_ahead: Forecast horizon (default: 24 hours)
    
    Returns:
        Prediction data or None if not available
        {
            "entity_id": "light.living_room",
            "generated_at": "2026-01-09T10:00:00",
            "predictions": [
                {
                    "timestamp": "2026-01-09T18:00:00",
                    "hour": 18,
                    "predicted_state": 1,
                    "predicted_state_label": "on",
                    "probability_on": 0.89,
                    "confidence": 0.89
                },
                ...
            ]
        }
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/predict/{entity_id}",
                params={"hours_ahead": hours_ahead},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Got prediction for {entity_id}: {len(data.get('predictions', []))} hours")
                    return data
                elif resp.status == 404:
                    logger.info(f"No model available for {entity_id}")
                    return None
                else:
                    logger.warning(f"Prediction failed for {entity_id}: {resp.status}")
                    return None
                    
    except asyncio.TimeoutError:
        logger.warning(f"Prediction timeout for {entity_id}")
        return None
    except Exception as e:
        logger.error(f"Prediction error for {entity_id}: {e}")
        return None


async def get_prediction_summary(entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Get human-readable prediction summary.
    
    Returns:
        {
            "entity_id": "light.living_room",
            "high_probability_hours": [18, 19, 20, 21, 22],
            "low_probability_hours": [0, 1, 2, 3, 4, 5],
            "next_state_change": {
                "time": "18:00",
                "to": "on",
                "confidence": 0.89
            },
            "overall_confidence": 0.86
        }
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/predict/{entity_id}/summary",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
                
    except Exception as e:
        logger.error(f"Summary error for {entity_id}: {e}")
        return None


async def find_next_state_change(entity_id: str, target_state: str = "on", hours_ahead: int = 24) -> Optional[Dict[str, Any]]:
    """
    Find the next time an entity is predicted to change to a target state.
    
    Useful for timing automations:
    - "When will light.kitchen turn on?" → "18:23 with 89% confidence"
    - "When will motion sensor go inactive?" → "23:45 with 75% confidence"
    
    Args:
        entity_id: Device to check
        target_state: State to look for ("on" or "off")
        hours_ahead: How far to look (default: 24 hours)
    
    Returns:
        {
            "entity_id": "light.living_room",
            "target_state": "on",
            "predicted_time": "2026-01-09T18:23:00",
            "hour": 18,
            "confidence": 0.89,
            "hours_from_now": 8.5
        }
    """
    prediction = await get_prediction(entity_id, hours_ahead)
    
    if not prediction or not prediction.get('predictions'):
        return None
    
    now = datetime.now()
    
    # Find first occurrence of target state with high confidence
    for pred in prediction['predictions']:
        if pred['predicted_state_label'] == target_state and pred['confidence'] > 0.7:
            pred_time = datetime.fromisoformat(pred['timestamp'])
            hours_from_now = (pred_time - now).total_seconds() / 3600
            
            return {
                "entity_id": entity_id,
                "target_state": target_state,
                "predicted_time": pred['timestamp'],
                "hour": pred['hour'],
                "confidence": pred['confidence'],
                "hours_from_now": round(hours_from_now, 1)
            }
    
    return None


async def check_if_model_exists(entity_id: str) -> bool:
    """
    Check if a prediction model exists for this entity.
    
    Returns:
        True if model exists, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/models/{entity_id}",
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                return resp.status == 200
                
    except Exception:
        return False


async def get_predicted_active_hours(entity_id: str, threshold: float = 0.8) -> List[int]:
    """
    Get hours when entity is predicted to be active/on.
    
    Args:
        entity_id: Device to check
        threshold: Probability threshold (default: 0.8 = 80%)
    
    Returns:
        List of hours (0-23) when device will likely be active
        Example: [18, 19, 20, 21, 22] (6pm-10pm)
    """
    prediction = await get_prediction(entity_id, hours_ahead=24)
    
    if not prediction or not prediction.get('predictions'):
        return []
    
    active_hours = []
    for pred in prediction['predictions']:
        if pred['probability_on'] >= threshold:
            active_hours.append(pred['hour'])
    
    return active_hours


async def enhance_pattern_with_predictions(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a detected pattern with predictive insights.
    
    This is the main function the Proactive Engine should call.
    It takes a pattern and adds prediction-based context.
    
    Args:
        pattern: Pattern detected by Proactive Engine
        {
            "description": "User turns on kitchen lights every evening",
            "entities": ["light.kitchen_roof_1", "light.kitchen_roof_2"],
            "frequency": "daily",
            "confidence": 0.85
        }
    
    Returns:
        Enhanced pattern with predictions:
        {
            ...original pattern...,
            "predictions": {
                "light.kitchen_roof_1": {
                    "next_on_time": "18:23",
                    "confidence": 0.89,
                    "typical_on_hours": [18, 19, 20, 21, 22]
                },
                ...
            },
            "suggested_timing": {
                "trigger_time": "18:20",
                "reason": "Light predicted to turn on at 18:23 - trigger 3 min early"
            }
        }
    """
    enhanced = pattern.copy()
    enhanced['predictions'] = {}
    
    entities = pattern.get('entities', [])
    
    for entity_id in entities:
        # Check if prediction exists
        has_model = await check_if_model_exists(entity_id)
        if not has_model:
            continue
        
        # Get prediction summary
        summary = await get_prediction_summary(entity_id)
        if not summary:
            continue
        
        # Find next state change
        next_on = await find_next_state_change(entity_id, target_state="on")
        next_off = await find_next_state_change(entity_id, target_state="off")
        
        # Get active hours
        active_hours = await get_predicted_active_hours(entity_id)
        
        enhanced['predictions'][entity_id] = {
            "has_model": True,
            "next_on_time": next_on['predicted_time'] if next_on else None,
            "next_on_confidence": next_on['confidence'] if next_on else None,
            "next_off_time": next_off['predicted_time'] if next_off else None,
            "typical_on_hours": active_hours,
            "overall_confidence": summary.get('overall_confidence')
        }
    
    # Suggest optimal timing if we have predictions
    if enhanced['predictions']:
        enhanced['suggested_timing'] = _calculate_optimal_timing(enhanced['predictions'])
    
    logger.info(f"Enhanced pattern with predictions for {len(enhanced['predictions'])} entities")
    
    return enhanced


def _calculate_optimal_timing(predictions: Dict[str, Dict]) -> Dict[str, str]:
    """
    Calculate optimal timing for automation based on predictions.
    
    Strategy:
    - If next state change predicted, trigger 3-5 minutes before
    - If multiple entities, use earliest predicted time
    - Only suggest if confidence > 75%
    """
    earliest_time = None
    earliest_confidence = 0
    reason = "No clear pattern"
    
    for entity_id, pred in predictions.items():
        if pred.get('next_on_time') and pred.get('next_on_confidence', 0) > 0.75:
            pred_time = datetime.fromisoformat(pred['next_on_time'])
            
            if earliest_time is None or pred_time < earliest_time:
                earliest_time = pred_time
                earliest_confidence = pred['next_on_confidence']
    
    if earliest_time:
        # Trigger 3 minutes before predicted time
        trigger_time = earliest_time - timedelta(minutes=3)
        reason = f"Predicted state change at {earliest_time.strftime('%H:%M')} (confidence: {earliest_confidence:.0%})"
        
        return {
            "trigger_time": trigger_time.strftime("%H:%M"),
            "reason": reason,
            "confidence": earliest_confidence
        }
    
    return {
        "trigger_time": None,
        "reason": reason,
        "confidence": 0
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test prediction fetching
        entity_id = "light.kitchen_roof_1"
        
        print(f"Testing prediction for {entity_id}...")
        
        # Get full prediction
        prediction = await get_prediction(entity_id)
        if prediction:
            print(f"✓ Got {len(prediction['predictions'])} hourly predictions")
        
        # Get summary
        summary = await get_prediction_summary(entity_id)
        if summary:
            print(f"✓ High probability hours: {summary['high_probability_hours']}")
        
        # Find next ON time
        next_on = await find_next_state_change(entity_id, "on")
        if next_on:
            print(f"✓ Next ON predicted at {next_on['hour']}:00 (confidence: {next_on['confidence']:.0%})")
        
        # Test pattern enhancement
        pattern = {
            "description": "Kitchen lights turn on every evening",
            "entities": [entity_id],
            "confidence": 0.85
        }
        
        enhanced = await enhance_pattern_with_predictions(pattern)
        print(f"\n✓ Enhanced pattern:")
        print(f"  Suggested timing: {enhanced.get('suggested_timing')}")
    
    asyncio.run(test())