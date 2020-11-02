module ReceptiveField.Pixel2retina where

import Prelude

import ReceptiveField.Main (render, int2img, imgPercentile)
import FRP.Event.Mouse (Mouse, getMouse)
import FRP.Behavior.Mouse as Mouse
import Web.DOM.NonElementParentNode (getElementById)
import Web.HTML (window)
import Web.HTML.HTMLDocument (toDocument)
import Web.HTML.Window (document)
import Data.Maybe (Maybe(..), fromJust, maybe)
import Global (readFloat)
import Web.DOM.Element (getAttribute)
import Web.DOM.Document (toNonElementParentNode)
import Partial.Unsafe (unsafePartial)
import Effect (Effect)
import Effect.Console (log)
import FRP.Behavior (Behavior, animate)

scene :: Mouse -> { w :: Number, h :: Number } -> Behavior String
scene mouse { w, h } = maybe "/pixel2retina/16_16.png"
  (\{x, y} -> int2img "/pixel2retina/" (widthPerc x w) (heightPerc y h) )
  <$> (Mouse.position mouse) where
    heightPerc = imgPercentile 64
    widthPerc = imgPercentile 64

main :: Effect Unit
main = do
  mouse <- getMouse
  win ← window
  doc ← document win
  maybeElement ← getElementById "retina2pixel" $ toNonElementParentNode $ toDocument  doc
  case maybeElement of
    Nothing → log "couldn't find img by ID"
    Just elem → do
      mH <- getAttribute "height" elem
      let h = readFloat (unsafePartial fromJust (mH))
      mW <- getAttribute "width" elem
      let w  = readFloat (unsafePartial (fromJust mW))
      _ <- animate (scene mouse {w, h}) (render elem)
      pure unit
  pure unit
