package com.bug423.inertiallab.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val Blue = Color(0xFF2672FF)
val Cyan = Color(0xFF28C4E8)
val Mint = Color(0xFF42D3A5)
val Coral = Color(0xFFFF6B65)
val Ink = Color(0xFF13213A)
val SubtleInk = Color(0xFF5D6B82)

private val LightColors = lightColorScheme(
    primary = Blue,
    onPrimary = Color.White,
    secondary = Cyan,
    tertiary = Mint,
    background = Color(0xFFF3F7FD),
    onBackground = Ink,
    surface = Color.White,
    onSurface = Ink,
    surfaceVariant = Color(0xFFE8EFF8),
    onSurfaceVariant = SubtleInk,
    error = Coral,
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF76A7FF),
    onPrimary = Color(0xFF002B69),
    secondary = Color(0xFF71D7F0),
    tertiary = Color(0xFF6EE7BC),
    background = Color(0xFF07101F),
    onBackground = Color(0xFFF2F6FF),
    surface = Color(0xFF101B2C),
    onSurface = Color(0xFFF2F6FF),
    surfaceVariant = Color(0xFF1C2A3F),
    onSurfaceVariant = Color(0xFFB5C1D5),
    error = Color(0xFFFF8A86),
)

private val LabTypography = androidx.compose.material3.Typography(
    displaySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 36.sp,
        lineHeight = 42.sp,
        letterSpacing = (-1.1).sp,
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 26.sp,
        lineHeight = 32.sp,
        letterSpacing = (-0.5).sp,
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 20.sp,
        lineHeight = 26.sp,
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp,
        lineHeight = 22.sp,
    ),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 14.sp),
)

@Composable
fun InertialLabTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = LabTypography,
        content = content,
    )
}
