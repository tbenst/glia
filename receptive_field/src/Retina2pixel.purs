module ReceptiveField.Retina2pixel where

import Prelude

import ReceptiveField.Main (render, int2img, imgPercentile, preloadImages)
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
import Effect.Aff (launchAff_)

-- addition of 'glia/' breaks local development but needed for gh-pages :/
scene :: Mouse -> { w :: Number, h :: Number } -> Behavior String
scene mouse { w, h } = maybe "retina2pixel/16_16.png"
  (\{x, y} -> int2img "retina2pixel/" (heightPerc y h) (widthPerc x w) )
  <$> (Mouse.position mouse) where
    heightPerc = imgPercentile 32
    widthPerc = imgPercentile 32

main :: Effect Unit
main = do
  mouse <- getMouse
  win ← window
  doc ← document win
  maybeElement ← getElementById "retina2pixel" $ toNonElementParentNode $ toDocument  doc
  _ <- launchAff_ $ preloadImages "retina2pixel/" 32 32
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
